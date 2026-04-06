import subprocess
from typing import Any, Dict, List, Optional, Tuple

from ..core.config import PATHFINDER_TEMP_DIR, CPP_EXE_PATH


def write_astar_input_file(file_path, rows, cols, start, goal, grid):
    with file_path.open("w", encoding="utf-8") as file:
        file.write(f"{rows} {cols}\n")
        file.write(f"{start[0]} {start[1]}\n")
        file.write(f"{goal[0]} {goal[1]}\n")
        for row in grid:
            file.write(" ".join(str(cell) for cell in row) + "\n")


def read_astar_output_file(file_path):
    if not file_path.exists():
        return {"success": False, "path": [], "message": "Pathfinder output file was not created."}

    lines = [
        line.strip()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not lines:
        return {"success": False, "path": [], "message": "Pathfinder output file was empty."}

    if lines[0] == "NO_PATH":
        return {"success": False, "path": [], "message": "No valid path could be found."}

    if lines[0] != "PATH_FOUND":
        return {"success": False, "path": [], "message": "Unexpected output format returned by C++ pathfinder."}

    try:
        path_count = int(lines[1])
    except (ValueError, IndexError):
        return {"success": False, "path": [], "message": "Path length in output file was invalid."}

    path = []
    for i in range(path_count):
        line_index = i + 2
        if line_index >= len(lines):
            return {"success": False, "path": [], "message": "Output file ended before the full path was read."}
        try:
            row, col = map(int, lines[line_index].split())
        except ValueError:
            return {"success": False, "path": [], "message": "A path coordinate in the output file was invalid."}
        path.append([row, col])

    return {"success": True, "path": path, "message": "Path found successfully."}


def run_astar_process(camera_id, rows, cols, start, goal, grid):
    input_file = PATHFINDER_TEMP_DIR / f"{camera_id}_input.txt"
    output_file = PATHFINDER_TEMP_DIR / f"{camera_id}_output.txt"

    write_astar_input_file(input_file, rows, cols, start, goal, grid)

    if not CPP_EXE_PATH.exists():
        return {"success": False, "path": [], "message": f"C++ executable was not found at: {CPP_EXE_PATH}"}

    try:
        result = subprocess.run(
            [str(CPP_EXE_PATH), str(input_file), str(output_file)],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "path": [], "message": "C++ pathfinder timed out after 10 seconds."}
    except Exception as e:
        return {"success": False, "path": [], "message": f"Failed to launch C++ pathfinder: {e}"}

    if result.returncode != 0:
        error_text = result.stderr.strip() or result.stdout.strip() or "Unknown C++ pathfinder error."
        return {"success": False, "path": [], "message": f"C++ exited with code {result.returncode}: {error_text}"}

    return read_astar_output_file(output_file)


def is_spot_available(spot: Dict[str, Any]) -> bool:
    if spot.get("occupied") is False:
        return True
    if spot.get("available") is True:
        return True
    if spot.get("is_available") is True:
        return True
    status_value = spot.get("status")
    if isinstance(status_value, str) and status_value.lower().strip() in ["available", "free", "vacant"]:
        return True
    return False


def point_in_polygon(x: float, y: float, polygon: List[Dict[str, float]]) -> bool:
    if not polygon or len(polygon) < 3:
        return False

    inside = False
    j = len(polygon) - 1

    for i in range(len(polygon)):
        xi, yi = float(polygon[i]["x"]), float(polygon[i]["y"])
        xj, yj = float(polygon[j]["x"]), float(polygon[j]["y"])

        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) if (yj - yi) != 0 else 1e-9) + xi
        )
        if intersects:
            inside = not inside
        j = i

    return inside


def sample_points_for_grid_cell(cell: List[int], rows: int, cols: int) -> List[Tuple[float, float]]:
    row, col = cell
    left, right = col / cols, (col + 1) / cols
    top, bottom = row / rows, (row + 1) / rows
    mid_x, mid_y = (left + right) / 2, (top + bottom) / 2
    inset_x, inset_y = (right - left) * 0.2, (bottom - top) * 0.2

    return [
        (mid_x, mid_y),
        (left + inset_x, top + inset_y),
        (right - inset_x, top + inset_y),
        (left + inset_x, bottom - inset_y),
        (right - inset_x, bottom - inset_y),
    ]


def find_spot_for_grid_cell(
    cell: List[int], rows: int, cols: int, spots: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    for idx, spot in enumerate(spots):
        polygon = spot.get("polygon", [])
        for px, py in sample_points_for_grid_cell(cell, rows, cols):
            if point_in_polygon(px, py, polygon):
                return spot
    return None


def get_available_parking_cells(
    parking_spaces: List[List[int]], rows: int, cols: int, spots_doc: Dict[str, Any]
) -> List[List[int]]:
    spots = spots_doc.get("spots", [])
    available_cells = []

    for cell in parking_spaces:
        matched_spot = find_spot_for_grid_cell(cell, rows, cols, spots)
        if matched_spot and is_spot_available(matched_spot):
            available_cells.append(cell)

    return available_cells


def find_nearest_available_path(camera_id, rows, cols, start, grid, parking_spaces, spots_doc):
    available_cells = get_available_parking_cells(
        parking_spaces=parking_spaces, rows=rows, cols=cols, spots_doc=spots_doc
    )

    if not available_cells:
        return {"success": False, "path": [], "goal": None, "message": "No available parking spaces were found."}

    best_result = None
    best_goal = None
    best_path_length = None

    for goal in available_cells:
        result = run_astar_process(
            camera_id=f"{camera_id}_{goal[0]}_{goal[1]}",
            rows=rows, cols=cols, start=start, goal=goal, grid=grid,
        )

        if not result.get("success"):
            continue

        current_path = result.get("path", [])
        if not current_path or len(current_path) < 2:
            continue

        if best_result is None or len(current_path) < best_path_length:
            best_result = result
            best_goal = goal
            best_path_length = len(current_path)

    if best_result is None:
        return {"success": False, "path": [], "goal": None, "message": "No valid route was found to any available parking space."}

    return {"success": True, "path": best_result["path"], "goal": best_goal, "message": "Nearest available parking space routed successfully."}
