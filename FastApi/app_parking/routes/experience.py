import asyncio
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import json
import shutil

import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from motor.motor_asyncio import AsyncIOMotorDatabase

from ..services.astar import find_nearest_available_path
from ..services.yolo import analyze_image_with_spots
from ..core.config import OUTPUT_DIR, UPLOAD_DIR, GRID_CONFIG_DIR, CAPTURE_DIR
from ..core.db import get_db

router = APIRouter(prefix="/api/experience", tags=["Experience"])


def get_grid_file_path(camera_id: str):
    return GRID_CONFIG_DIR / f"{camera_id}.json"


@router.post("/run")
async def run_experience(
    file: UploadFile = File(...),
    camera_id: str = Form("cam1"),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="no file uploaded")
    if not file.content_type or not file.content_type.startswith("video"):
        raise HTTPException(status_code=400, detail="file must be a video")

    spots_doc = await db.spot_configs.find_one(
        {"camera_id": camera_id, "status": "published"}, sort=[("version", -1)]
    )
    if not spots_doc:
        raise HTTPException(status_code=400, detail="no published spot config for this camera_id")

    grid_file = get_grid_file_path(camera_id)
    if not grid_file.exists():
        raise HTTPException(status_code=400, detail="no saved grid config for this camera_id")

    grid_data = json.loads(grid_file.read_text(encoding="utf-8"))
    start = grid_data.get("start")
    parking_spaces = grid_data.get("parking_spaces", [])

    if not start or len(start) != 2:
        raise HTTPException(status_code=400, detail="grid config is missing a valid start cell")

    job_id = str(uuid4())
    ext = Path(file.filename).suffix.lower() or ".mp4"
    upload_path = UPLOAD_DIR / f"{job_id}{ext}"

    loop = asyncio.get_event_loop()
    file_bytes = await file.read()
    await loop.run_in_executor(None, upload_path.write_bytes, file_bytes)

    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = job_dir / "snapshot.jpg"

    def extract_and_analyse():
        cap = cv2.VideoCapture(str(upload_path))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("could not read the first video frame")
        cv2.imwrite(str(snapshot_path), frame)
        return analyze_image_with_spots(snapshot_path, spots_doc)

    try:
        analysis = await loop.run_in_executor(None, extract_and_analyse)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    status_by_id = {s["id"]: s["status"] for s in analysis["spots"]}
    merged_spots = [
        {**s, "status": status_by_id.get(s.get("id", ""), "occupied")}
        for s in spots_doc.get("spots", [])
    ]

    route_result = await loop.run_in_executor(
        None,
        lambda: find_nearest_available_path(
            camera_id=camera_id,
            rows=grid_data["rows"],
            cols=grid_data["cols"],
            start=start,
            grid=grid_data["grid"],
            parking_spaces=parking_spaces,
            spots_doc={"spots": merged_spots},
        ),
    )

    await db.jobs.insert_one({
        "_id": job_id,
        "camera_id": camera_id,
        "status": "snapshot_analyzed",
        "created_at": datetime.utcnow(),
        "input_path": str(upload_path),
        "snapshot_file": str(snapshot_path.name),
        "available": analysis["available"],
        "occupied": analysis["occupied"],
        "error": None,
    })

    return {
        "job_id": job_id,
        "camera_id": camera_id,
        "snapshot_url": f"/api/jobs/{job_id}/snapshot",
        "available": analysis["available"],
        "occupied": analysis["occupied"],
        "spots": analysis["spots"],
        "path": route_result.get("path", []),
        "goal": route_result.get("goal"),
        "start": start,
        "route_success": route_result.get("success", False),
        "route_message": route_result.get("message", ""),
    }


@router.post("/run-from-capture")
async def run_from_capture(
    camera_id: str = Form("cam1"),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    capture_doc = await db.captures.find_one({"_id": camera_id})
    if not capture_doc:
        raise HTTPException(status_code=404, detail="no snapshot uploaded yet for this camera")

    capture_path = CAPTURE_DIR / capture_doc["file"]
    if not capture_path.exists():
        raise HTTPException(status_code=404, detail="snapshot file missing on disk")

    spots_doc = await db.spot_configs.find_one(
        {"camera_id": camera_id, "status": "published"}, sort=[("version", -1)]
    )
    if not spots_doc:
        raise HTTPException(status_code=400, detail="no published spot config for this camera_id")

    grid_file = get_grid_file_path(camera_id)
    if not grid_file.exists():
        raise HTTPException(status_code=400, detail="no saved grid config for this camera_id")

    grid_data = json.loads(grid_file.read_text(encoding="utf-8"))
    start = grid_data.get("start")
    parking_spaces = grid_data.get("parking_spaces", [])

    if not start or len(start) != 2:
        raise HTTPException(status_code=400, detail="grid config is missing a valid start cell")

    job_id = str(uuid4())
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = job_dir / "snapshot.jpg"

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, shutil.copy2, str(capture_path), str(snapshot_path))

    try:
        analysis = await loop.run_in_executor(
            None, lambda: analyze_image_with_spots(snapshot_path, spots_doc)
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    status_by_id = {s["id"]: s["status"] for s in analysis["spots"]}
    merged_spots = [
        {**s, "status": status_by_id.get(s.get("id", ""), "occupied")}
        for s in spots_doc.get("spots", [])
    ]

    route_result = await loop.run_in_executor(
        None,
        lambda: find_nearest_available_path(
            camera_id=camera_id,
            rows=grid_data["rows"],
            cols=grid_data["cols"],
            start=start,
            grid=grid_data["grid"],
            parking_spaces=parking_spaces,
            spots_doc={"spots": merged_spots},
        ),
    )

    await db.jobs.insert_one({
        "_id": job_id,
        "camera_id": camera_id,
        "status": "snapshot_analyzed",
        "created_at": datetime.utcnow(),
        "input_path": str(capture_path),
        "snapshot_file": str(snapshot_path.name),
        "available": analysis["available"],
        "occupied": analysis["occupied"],
        "error": None,
    })

    return {
        "job_id": job_id,
        "camera_id": camera_id,
        "snapshot_url": f"/api/jobs/{job_id}/snapshot",
        "available": analysis["available"],
        "occupied": analysis["occupied"],
        "spots": analysis["spots"],
        "path": route_result.get("path", []),
        "goal": route_result.get("goal"),
        "start": start,
        "route_success": route_result.get("success", False),
        "route_message": route_result.get("message", ""),
    }
