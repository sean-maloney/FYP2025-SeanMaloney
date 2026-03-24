from pathlib import Path
from datetime import datetime
import shutil
import mimetypes

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from fastapi.responses import FileResponse
from motor.motor_asyncio import AsyncIOMotorDatabase

from .config import CAPTURE_DIR
from .db import get_db
from .yolo_service import analyze_image_with_spots

router = APIRouter(tags=["captures"])


@router.post("/api/pi/upload-snapshot")
async def upload_snapshot_from_pi(
    file: UploadFile = File(...),
    camera_id: str = Form("cam1"),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="no file uploaded")

    ext = Path(file.filename).suffix.lower() or ".jpg"
    if ext not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail="file must be jpg, jpeg, or png")

    save_path = CAPTURE_DIR / f"{camera_id}_latest{ext}"

    with save_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    await db.captures.update_one(
        {"_id": camera_id},
        {
            "$set": {
                "_id": camera_id,
                "camera_id": camera_id,
                "file": save_path.name,
                "updated_at": datetime.utcnow(),
            }
        },
        upsert=True,
    )

    return {"status": "ok", "camera_id": camera_id, "image_url": f"/api/cameras/{camera_id}/image"}


@router.post("/api/cameras/{camera_id}/capture")
async def capture_current_camera_image(
    camera_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    doc = await db.captures.find_one({"_id": camera_id})
    if not doc:
        raise HTTPException(status_code=404, detail="no snapshot uploaded yet for this camera")

    return {
        "camera_id": camera_id,
        "image_url": f"/api/cameras/{camera_id}/image",
        "updated_at": doc.get("updated_at"),
    }


@router.get("/api/cameras/{camera_id}/image", response_class=FileResponse)
async def get_current_camera_image(
    camera_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    doc = await db.captures.find_one({"_id": camera_id})
    if not doc:
        raise HTTPException(status_code=404, detail="no snapshot uploaded yet for this camera")

    image_path = CAPTURE_DIR / doc["file"]
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="snapshot missing on disk")

    media_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    return FileResponse(str(image_path), media_type=media_type, filename=image_path.name)


@router.post("/api/cameras/{camera_id}/refresh")
async def refresh_camera_status(
    camera_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    capture_doc = await db.captures.find_one({"_id": camera_id})
    if not capture_doc:
        raise HTTPException(status_code=404, detail="no snapshot uploaded yet for this camera")

    image_path = CAPTURE_DIR / capture_doc["file"]
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="snapshot missing on disk")

    spots_doc = await db.spot_configs.find_one(
        {"camera_id": camera_id, "status": "published"},
        sort=[("version", -1)],
    )
    if not spots_doc:
        raise HTTPException(status_code=400, detail="no published spot config for this camera_id")

    try:
        result = analyze_image_with_spots(
            input_image=image_path,
            spots_doc=spots_doc,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "camera_id": camera_id,
        "image_url": f"/api/cameras/{camera_id}/image",
        "available": int(result["available"]),
        "occupied": int(result["occupied"]),
        "spots": result["spots"],
        "updated_at": capture_doc.get("updated_at"),
    }