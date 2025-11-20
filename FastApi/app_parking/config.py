from pathlib import Path

# this file lives in: FastApi/app_parking/config.py
BASE_DIR = Path(__file__).resolve().parent        # .../FastApi/app_parking
PROJECT_ROOT = BASE_DIR.parent.parent             # .../FYP-clean

# folders for uploads and outputs
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# path to trained YOLO weights
# 🔧 ADJUST THIS to match your real path
# if best.pt is under runs/train/... use "train"
# if it's under runs/detect/... use "detect"
YOLO_WEIGHTS_PATH = (
    PROJECT_ROOT
    / "runs"
    / "train"                     # or "detect"
    / "visdrone_parking_detector5"  # folder name under runs/train or runs/detect
    / "weights"
    / "best.pt"
)

YOLO_DEVICE = "cuda"  # or "cpu" if you want to force CPU
