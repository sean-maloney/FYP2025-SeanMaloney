import json

from fastapi import APIRouter, HTTPException

from .config import GRID_CONFIG_DIR
from .astar_service import run_astar_process


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