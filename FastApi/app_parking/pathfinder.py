from fastapi import APIRouter, HTTPException
from app.models.grid_models import GridConfigRequest, PathRequest, PathResponse
from app.services.grid_service import save_grid_config, load_grid_config
from app.services.astar_service import run_astar_process

router = APIRouter()


@router.post("/grid/save")
def save_grid(data: GridConfigRequest):
    save_grid_config(data.model_dump())
    return {"message": "Grid config saved successfully"}


@router.get("/grid/{camera_id}")
def get_grid(camera_id: str):
    data = load_grid_config(camera_id)

    if data is None:
        raise HTTPException(status_code=404, detail="Grid config not found")

    return data


@router.post("/path/run", response_model=PathResponse)
def run_path(data: PathRequest):
    grid_data = load_grid_config(data.camera_id)

    if grid_data is None:
        raise HTTPException(status_code=404, detail="Grid config not found")

    result = run_astar_process(
        camera_id=data.camera_id,
        rows=grid_data["rows"],
        cols=grid_data["cols"],
        start=grid_data["start"],
        goal=data.goal,
        grid=grid_data["grid"]
    )

    return result