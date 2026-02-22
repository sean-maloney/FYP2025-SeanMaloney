from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4
from datetime import datetime
import shutil

import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from motor.motor_asyncio import AsyncIOMotorDatabase

from .config import UPLOAD_DIR, OUTPUT_DIR
from .db import connect_mongo, close_mongo, get_db
from .spots_routes import router as spots_router
from .yolo_service import load_yolo_model, run_inference_with_spots


@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    load_yolo_model()
    await connect_mongo()
    yield
    await close_mongo()


app = FastAPI(
    title="Parking Detector API",
    description="Upload a video, draw spots, then run inference.",
    lifespan=lifespan,
)

app.include_router(spots_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/videos", status_code=status.HTTP_201_CREATED)
async def upload_video(
    file: UploadFile = File(...),
    camera_id: str = Form("cam1"),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    job_id = str(uuid4())

    if not file.filename:
        raise HTTPException(status_code=400, detail="no file uploaded")

    if not file.content_type or not file.content_type.startswith("video"):
        raise HTTPException(status_code=400, detail="file must be a video")

    ext = Path(file.filename).suffix.lower() or ".mp4"
    upload_path = UPLOAD_DIR / f"{job_id}{ext}"

    with upload_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    await db.jobs.insert_one(
        {
            "_id": job_id,
            "camera_id": camera_id,
            "status": "uploaded",
            "created_at": datetime.utcnow(),
            "input_path": str(upload_path),
            "output_file": None,
            "snapshot_file": None,
            "available": None,
            "occupied": None,
            "error": None,
        }
    )

    return {"job_id": job_id, "camera_id": camera_id}


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str, db: AsyncIOMotorDatabase = Depends(get_db)):
    job = await db.jobs.find_one({"_id": job_id})
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    job["_id"] = str(job["_id"])
    return job


@app.get("/api/jobs/{job_id}/snapshot")
async def job_snapshot(job_id: str, db: AsyncIOMotorDatabase = Depends(get_db)):
    job = await db.jobs.find_one({"_id": job_id})
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    job_dir = OUTPUT_DIR / job_id
    snap_path = job_dir / "snapshot.jpg"

    if not snap_path.exists():
        input_path = job.get("input_path")
        if not input_path:
            raise HTTPException(status_code=500, detail="job missing input_path")

        cap = cv2.VideoCapture(str(input_path))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise HTTPException(status_code=500, detail="could not read video frame")

        cv2.imwrite(str(snap_path), frame)

        await db.jobs.update_one(
            {"_id": job_id},
            {"$set": {"snapshot_file": str(snap_path.name)}},
        )

    return FileResponse(str(snap_path), media_type="image/jpeg", filename="snapshot.jpg")


@app.post("/api/jobs/{job_id}/run")
async def run_job(job_id: str, db: AsyncIOMotorDatabase = Depends(get_db)):
    job = await db.jobs.find_one({"_id": job_id})
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    if job.get("status") == "processing":
        return {"job_id": job_id, "status": "processing"}

    input_path = job.get("input_path")
    camera_id = job.get("camera_id")
    if not input_path or not camera_id:
        raise HTTPException(status_code=500, detail="job missing input_path or camera_id")

    spots_doc = await db.spot_configs.find_one(
        {"camera_id": camera_id, "status": "published"},
        sort=[("version", -1)],
    )
    if not spots_doc:
        raise HTTPException(status_code=400, detail="no published spot config for this camera_id")

    await db.jobs.update_one({"_id": job_id}, {"$set": {"status": "processing", "error": None}})

    try:
        job_dir = OUTPUT_DIR / job_id
        out_path, available, occupied = run_inference_with_spots(
            input_video=Path(input_path),
            output_dir=job_dir,
            spots_doc=spots_doc,
        )

        await db.jobs.update_one(
            {"_id": job_id},
            {"$set": {
                "status": "done",
                "output_file": out_path.name,
                "available": int(available),
                "occupied": int(occupied),
            }},
        )
        return {"job_id": job_id, "status": "done", "output_file": out_path.name, "available": available, "occupied": occupied}

    except Exception as e:
        await db.jobs.update_one(
            {"_id": job_id},
            {"$set": {"status": "failed", "error": str(e)}},
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/videos/{job_id}", response_class=FileResponse)
async def get_video(job_id: str, db: AsyncIOMotorDatabase = Depends(get_db)):
    job = await db.jobs.find_one({"_id": job_id})
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    if job.get("status") != "done" or not job.get("output_file"):
        raise HTTPException(status_code=409, detail=f"video not ready (status={job.get('status')})")

    video_path = (OUTPUT_DIR / job_id) / job["output_file"]
    if not video_path.exists():
        raise HTTPException(status_code=500, detail="output file missing on disk")

    return FileResponse(
        str(video_path),
        media_type="video/mp4",
        filename=video_path.name,
        headers={"Content-Disposition": f'inline; filename="{video_path.name}"'},
    )
