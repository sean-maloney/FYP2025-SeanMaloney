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
        return {"success": False, "path": [], "message": "output file missing"}

    lines = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    if not lines:
        return {"success": False, "path": [], "message": "output file empty"}

    if lines[0] == "NO_PATH":
        return {"success": False, "path": [], "message": "no path found"}

    if lines[0] != "PATH_FOUND":
        return {"success": False, "path": [], "message": "unexpected output format"}

    path_count = int(lines[1])
    path = []

    for i in range(path_count):
        row, col = map(int, lines[i + 2].split())
        path.append([row, col])

    return {"success": True, "path": path, "message": "path found"}


def run_astar_process(camera_id, rows, cols, start, goal, grid):
    input_file = PATHFINDER_TEMP_DIR / f"{camera_id}_input.txt"
    output_file = PATHFINDER_TEMP_DIR / f"{camera_id}_output.txt"

    write_astar_input_file(input_file, rows, cols, start, goal, grid)

    if not CPP_EXE_PATH.exists():
        return {
            "success": False,
            "path": [],
            "message": f"cpp executable not found: {CPP_EXE_PATH}",
        }

    result = subprocess.run(
        [str(CPP_EXE_PATH), str(input_file), str(output_file)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        error_text = result.stderr.strip() or result.stdout.strip() or "unknown cpp error"
        return {"success": False, "path": [], "message": error_text}

    return read_astar_output_file(output_file)