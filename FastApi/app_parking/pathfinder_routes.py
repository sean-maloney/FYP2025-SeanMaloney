import asyncio
import json
import shutil
from pathlib import Path
from uuid import uuid4

import cv2
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from motor.motor_asyncio import AsyncIOMotorDatabase

from .config import GRID_CONFIG_DIR, OUTPUT_DIR
from .db import get_db
from .astar_service import run_astar_process, find_nearest_available_path


router = APIRouter(prefix="/api/pathfinder", tags=["Pathfinder"])


def get_grid_file_path(camera_id: str):
    return GRID_CONFIG_DIR / f"{camera_id}.json"


def is_valid_cell(cell, rows, cols):
    if not isinstance(cell, list) or len(cell) != 2:
        return False

    row, col = cell

    if not isinstance(row, int) or not isinstance(col, int):
        return False

    return 0 <= row < rows and 0 <= col < cols


@router.post("/grid/save")
async def save_grid_config(payload: dict):
    camera_id = payload.get("camera_id")
    rows = payload.get("rows")
    cols = payload.get("cols")
    grid = payload.get("grid")
    start = payload.get("start")
    parking_spaces = payload.get("parking_spaces", [])

    if not camera_id:
        raise HTTPException(status_code=400, detail="camera_id is required")

    if not isinstance(rows, int) or rows <= 0:
        raise HTTPException(status_code=400, detail="rows must be a positive integer")

    if not isinstance(cols, int) or cols <= 0:
        raise HTTPException(status_code=400, detail="cols must be a positive integer")

    if not isinstance(grid, list) or len(grid) != rows:
        raise HTTPException(status_code=400, detail="grid row count does not match rows")

    for row in grid:
        if not isinstance(row, list) or len(row) != cols:
            raise HTTPException(status_code=400, detail="grid column count does not match cols")

    if start and not is_valid_cell(start, rows, cols):
        raise HTTPException(status_code=400, detail="start cell is invalid")

    for parking_cell in parking_spaces:
        if not is_valid_cell(parking_cell, rows, cols):
            raise HTTPException(status_code=400, detail="one or more parking space cells are invalid")

    file_path = get_grid_file_path(camera_id)
    file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {"message": "grid config saved", "camera_id": camera_id}


@router.get("/grid/{camera_id}")
async def load_grid_config(camera_id: str):
    file_path = get_grid_file_path(camera_id)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="grid config not found")

    return json.loads(file_path.read_text(encoding="utf-8"))


@router.post("/grid-source/video")
async def create_grid_source_from_video(
    file: UploadFile = File(...),
    camera_id: str = Form("cam1"),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="no file uploaded")

    if not file.content_type or not file.content_type.startswith("video"):
        raise HTTPException(status_code=400, detail="file must be a video")

    job_id = str(uuid4())
    ext = Path(file.filename).suffix.lower() or ".mp4"

    job_dir = OUTPUT_DIR / f"gridsetup_{job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / f"input{ext}"
    snapshot_path = job_dir / "snapshot.jpg"

    with input_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cap = cv2.VideoCapture(str(input_path))
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise HTTPException(status_code=500, detail="could not read the first frame from the uploaded video")

    cv2.imwrite(str(snapshot_path), frame)

    return {
        "camera_id": camera_id,
        "source_type": "video",
        "job_id": job_id,
        "image_url": f"/api/pathfinder/grid-source/video/{job_id}/image",
    }


@router.get("/grid-source/video/{job_id}/image")
async def get_grid_source_video_image(job_id: str):
    snapshot_path = OUTPUT_DIR / f"gridsetup_{job_id}" / "snapshot.jpg"

    if not snapshot_path.exists():
        raise HTTPException(status_code=404, detail="grid source image not found")

    from fastapi.responses import FileResponse
    return FileResponse(str(snapshot_path), media_type="image/jpeg", filename="snapshot.jpg")


@router.post("/run/{camera_id}")
async def run_pathfinder(camera_id: str, payload: dict):
    file_path = get_grid_file_path(camera_id)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="grid config not found")

    grid_data = json.loads(file_path.read_text(encoding="utf-8"))

    rows = grid_data["rows"]
    cols = grid_data["cols"]
    grid = grid_data["grid"]

    start = payload.get("start") or grid_data.get("start")
    goal = payload.get("goal")

    if not start:
        raise HTTPException(status_code=400, detail="start point is missing")

    if not goal:
        raise HTTPException(status_code=400, detail="goal point is missing")

    if not is_valid_cell(start, rows, cols):
        raise HTTPException(status_code=400, detail="start point is outside the grid")

    if not is_valid_cell(goal, rows, cols):
        raise HTTPException(status_code=400, detail="goal point is outside the grid")

    if grid[start[0]][start[1]] == 1:
        raise HTTPException(status_code=400, detail="start point is on a blocked cell")

    if grid[goal[0]][goal[1]] == 1:
        raise HTTPException(status_code=400, detail="goal point is on a blocked cell")

    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: run_astar_process(
            camera_id=camera_id,
            rows=rows,
            cols=cols,
            start=start,
            goal=goal,
            grid=grid,
        ),
    )

    return result


@router.post("/run-nearest/{camera_id}")
async def run_nearest_available_path(camera_id: str, payload: dict, db: AsyncIOMotorDatabase = Depends(get_db)):
    file_path = get_grid_file_path(camera_id)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="grid config not found")

    grid_data = json.loads(file_path.read_text(encoding="utf-8"))

    rows = grid_data["rows"]
    cols = grid_data["cols"]
    grid = grid_data["grid"]

    start = payload.get("start") or grid_data.get("start")
    if not start:
        raise HTTPException(status_code=400, detail="start point is missing")

    if not is_valid_cell(start, rows, cols):
        raise HTTPException(status_code=400, detail="start point is outside the grid")

    if grid[start[0]][start[1]] == 1:
        raise HTTPException(status_code=400, detail="start point is on a blocked cell")

    parking_spaces = grid_data.get("parking_spaces", [])
    if not parking_spaces:
        raise HTTPException(status_code=400, detail="no parking spaces mapped in grid config")

    for parking_cell in parking_spaces:
        if not is_valid_cell(parking_cell, rows, cols):
            raise HTTPException(status_code=400, detail="one or more mapped parking space cells are invalid")

    spots_doc = await db.spot_configs.find_one(
        {"camera_id": camera_id, "status": "published"},
        sort=[("version", -1)],
    )

    if not spots_doc:
        raise HTTPException(status_code=404, detail="no published spot config for this camera_id")

    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: find_nearest_available_path(
            camera_id=camera_id,
            rows=rows,
            cols=cols,
            start=start,
            grid=grid,
            parking_spaces=parking_spaces,
            spots_doc=spots_doc,
        ),
    )

    return result