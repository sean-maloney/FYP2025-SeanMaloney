from typing import List, Literal
from pydantic import BaseModel

CellType = Literal["empty", "road", "blocked", "start", "parking"]


class GridConfigRequest(BaseModel):
    camera_id: str
    rows: int
    cols: int
    grid: List[List[int]]
    start: List[int]
    parking_spaces: List[List[int]]


class PathRequest(BaseModel):
    camera_id: str
    goal: List[int]


class PathResponse(BaseModel):
    success: bool
    path: List[List[int]]
    message: str
