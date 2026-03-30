import json

from fastapi import APIRouter, HTTPException, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from .config import GRID_CONFIG_DIR
from .db import get_db
from .astar_service import run_astar_process, find_nearest_available_path


router = APIRouter(prefix="/api/pathfinder", tags=["Pathfinder"])


def get_grid_file_path(camera_id: str):
    return GRID_CONFIG_DIR / f"{camera_id}.json"


@router.post("/grid/save")
async def save_grid_config(payload: dict):
    camera_id = payload.get("camera_id")
    if not camera_id:
        raise HTTPException(status_code=400, detail="camera_id is required")

    file_path = get_grid_file_path(camera_id)
    file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {"message": "grid config saved", "camera_id": camera_id}


@router.get("/grid/{camera_id}")
async def load_grid_config(camera_id: str):
    file_path = get_grid_file_path(camera_id)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="grid config not found")

    return json.loads(file_path.read_text(encoding="utf-8"))


@router.post("/run/{camera_id}")
async def run_pathfinder(camera_id: str, payload: dict):
    file_path = get_grid_file_path(camera_id)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="grid config not found")

    grid_data = json.loads(file_path.read_text(encoding="utf-8"))

    start = payload.get("start") or grid_data.get("start")
    goal = payload.get("goal")

    if not start or len(start) != 2:
        raise HTTPException(status_code=400, detail="start point is missing")

    if not goal or len(goal) != 2:
        raise HTTPException(status_code=400, detail="goal point is missing")

    result = run_astar_process(
        camera_id=camera_id,
        rows=grid_data["rows"],
        cols=grid_data["cols"],
        start=start,
        goal=goal,
        grid=grid_data["grid"],
    )

    return result


@router.post("/run-nearest/{camera_id}")
async def run_nearest_available_path(camera_id: str, payload: dict, db: AsyncIOMotorDatabase = Depends(get_db)):
    file_path = get_grid_file_path(camera_id)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="grid config not found")

    grid_data = json.loads(file_path.read_text(encoding="utf-8"))

    start = payload.get("start") or grid_data.get("start")
    if not start or len(start) != 2:
        raise HTTPException(status_code=400, detail="start point is missing")

    parking_spaces = grid_data.get("parking_spaces", [])
    if not parking_spaces:
        raise HTTPException(status_code=400, detail="no parking spaces mapped in grid config")

    spots_doc = await db.spot_configs.find_one(
        {"camera_id": camera_id, "status": "published"},
        sort=[("version", -1)],
    )

    if not spots_doc:
        raise HTTPException(status_code=404, detail="no published spot config for this camera_id")

    result = find_nearest_available_path(
        camera_id=camera_id,
        rows=grid_data["rows"],
        cols=grid_data["cols"],
        start=start,
        grid=grid_data["grid"],
        parking_spaces=parking_spaces,
        spots_doc=spots_doc,
    )

    return result