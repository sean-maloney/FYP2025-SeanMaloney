import subprocess

from .config import PATHFINDER_TEMP_DIR, CPP_EXE_PATH


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

    lines = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    if not lines:
        return {"success": False, "path": [], "message": "Pathfinder output file was empty."}

    if lines[0] == "NO_PATH":
        return {"success": False, "path": [], "message": "No valid path could be found."}

    if lines[0] != "PATH_FOUND":
        return {"success": False, "path": [], "message": "Unexpected output format returned by C++ pathfinder."}

    if len(lines) < 2:
        return {"success": False, "path": [], "message": "Output file was missing the path length."}

    try:
        path_count = int(lines[1])
    except ValueError:
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
        return {
            "success": False,
            "path": [],
            "message": f"C++ executable was not found at: {CPP_EXE_PATH}",
        }

    result = subprocess.run(
        [str(CPP_EXE_PATH), str(input_file), str(output_file)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        error_text = result.stderr.strip() or result.stdout.strip() or "Unknown C++ pathfinder error."
        return {"success": False, "path": [], "message": error_text}

    return read_astar_output_file(output_file)


def is_spot_available(spot):
    if spot.get("occupied") is False:
        return True

    if spot.get("available") is True:
        return True

    if spot.get("is_available") is True:
        return True

    status_value = spot.get("status")
    if isinstance(status_value, str):
        status_value = status_value.lower().strip()
        if status_value in ["available", "free", "vacant"]:
            return True

    return False


def get_available_parking_cells(parking_spaces, spots_doc):
    spots = spots_doc.get("spots", [])
    available_cells = []

    for index, spot in enumerate(spots):
        if index >= len(parking_spaces):
            break

        if is_spot_available(spot):
            available_cells.append(parking_spaces[index])

    return available_cells


def find_nearest_available_path(camera_id, rows, cols, start, grid, parking_spaces, spots_doc):
    available_cells = get_available_parking_cells(parking_spaces, spots_doc)

    if not available_cells:
        return {
            "success": False,
            "path": [],
            "goal": None,
            "message": "No available parking spaces were found.",
        }

    best_result = None
    best_goal = None
    best_path_length = None

    for goal in available_cells:
        result = run_astar_process(
            camera_id=f"{camera_id}_{goal[0]}_{goal[1]}",
            rows=rows,
            cols=cols,
            start=start,
            goal=goal,
            grid=grid,
        )

        if not result.get("success"):
            continue

        current_path = result.get("path", [])
        current_length = len(current_path)

        if best_result is None or current_length < best_path_length:
            best_result = result
            best_goal = goal
            best_path_length = current_length

    if best_result is None:
        return {
            "success": False,
            "path": [],
            "goal": None,
            "message": "No valid route was found to any available parking space.",
        }

    return {
        "success": True,
        "path": best_result["path"],
        "goal": best_goal,
        "message": "Nearest available parking space routed successfully.",
    }