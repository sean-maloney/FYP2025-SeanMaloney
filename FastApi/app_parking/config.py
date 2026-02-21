from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YOLO_WEIGHTS_PATH = Path(os.getenv("YOLO_WEIGHTS_PATH", str(
    PROJECT_ROOT / "runs" / "detect" / "visdrone_parking_detector4" / "weights" / "best.pt"
)))

YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cuda")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "fyp_parking")
