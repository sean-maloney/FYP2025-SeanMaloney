from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
CAPTURE_DIR = BASE_DIR / "captures"

GRID_CONFIG_DIR = BASE_DIR / "grid_configs"
PATHFINDER_TEMP_DIR = BASE_DIR / "pathfinder_temp"
CPP_EXE_PATH = PROJECT_ROOT / "cpp_pathfinder" / "build" / "x64" / "Debug" / "AStarProject.exe"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
GRID_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
PATHFINDER_TEMP_DIR.mkdir(parents=True, exist_ok=True)

YOLO_WEIGHTS_PATH = Path(
    os.getenv(
        "YOLO_WEIGHTS_PATH",
        str(
            PROJECT_ROOT
            / "runs"
            / "detect"
            / "visdrone_parking_detector4"
            / "weights"
            / "best.pt"
        ),
    )
)

YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cuda")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "fyp_parking")