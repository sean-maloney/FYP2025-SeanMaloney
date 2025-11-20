from pathlib import Path
import cv2  # type: ignore
from ultralytics import YOLO  # type: ignore

from .config import YOLO_WEIGHTS_PATH, YOLO_DEVICE, OUTPUT_DIR

yolo_model: YOLO | None = None


def load_yolo_model():
    global yolo_model

    if not YOLO_WEIGHTS_PATH.exists():
        print(f"[WARN] YOLO weights not found at: {YOLO_WEIGHTS_PATH}")
        yolo_model = None
        return

    print(f"[INFO] Loading YOLO model from: {YOLO_WEIGHTS_PATH}")
    model = YOLO(str(YOLO_WEIGHTS_PATH))
    model.to(YOLO_DEVICE)
    yolo_model = model
    print("[INFO] YOLO model loaded")


def run_yolo_on_video(input_path: Path, job_name: str) -> Path:
    if yolo_model is None:
        raise RuntimeError("YOLO model not loaded")

    job_dir = OUTPUT_DIR / job_name
    job_dir.mkdir(parents=True, exist_ok=True)

    # run yolo – this will save a video (usually AVI) in job_dir
    yolo_model.predict(
        source=str(input_path),
        save=True,
        project=str(OUTPUT_DIR),
        name=job_name,
        show=False,
        vid_stride=1,
    )

    # check for any mp4 or avi files YOLO wrote
    mp4_files = list(job_dir.glob("*.mp4"))
    avi_files = list(job_dir.glob("*.avi"))

    if mp4_files:
        # yolo already wrote an mp4, just return it
        return mp4_files[0]

    if not avi_files:
        raise RuntimeError(f"No output video found in {job_dir}")

    avi_path = avi_files[0]
    mp4_path = job_dir / "output_fixed.mp4"
    convert_avi_to_mp4(avi_path, mp4_path)
    return mp4_path


def convert_avi_to_mp4(input_path: Path, output_path: Path):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise IOError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"[INFO] saved mp4 to {output_path}")
