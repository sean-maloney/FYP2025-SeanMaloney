import json
import os


GRID_FOLDER = "backend/grid_configs"


def ensure_grid_folder():
    os.makedirs(GRID_FOLDER, exist_ok=True)


def get_grid_file_path(camera_id: str) -> str:
    ensure_grid_folder()
    return os.path.join(GRID_FOLDER, f"{camera_id}.json")


def save_grid_config(data: dict):
    file_path = get_grid_file_path(data["camera_id"])

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_grid_config(camera_id: str):
    file_path = get_grid_file_path(camera_id)

    if not os.path.exists(file_path):
        return None

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)