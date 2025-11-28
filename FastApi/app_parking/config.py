from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

# folders for uploads and outputs
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#adjust this to your actual path
YOLO_WEIGHTS_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"                    
    / "visdrone_parking_detector5" 
    / "weights"
    / "best.pt"
)

YOLO_DEVICE = "cuda"  #uses gpu (i have nvidia so it's better) if not change it to "cpu"