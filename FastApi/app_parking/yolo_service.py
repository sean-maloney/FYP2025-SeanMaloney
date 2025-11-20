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

    # run yolo, saves video as avi, then stores it in job dir
    yolo_model.predict(
        source=str(input_path),
        save=True,
        project=str(OUTPUT_DIR),
        name=job_name,
        show=False,
        vid_stride=1,
    )

    # check for any mp4 or avi files from yolo
    mp4_files = list(job_dir.glob("*.mp4"))
    avi_files = list(job_dir.glob("*.avi"))

    if mp4_files:
        # yolo already wrote an mp4, just return it but this rarely happens if at all
        return mp4_files[0]

    #if it cant find avi or mp4 so no video was created and give an error
    if not avi_files:
        raise RuntimeError(f"No output video found in {job_dir}")

    avi_path = avi_files[0] #takes the list and gets the first one
    mp4_path = job_dir / "output_fixed.mp4" #builds path for new file.
    convert_avi_to_mp4(avi_path, mp4_path) #calls the conversion for avi to mp4
    return mp4_path #returns the path to the mp4


def convert_avi_to_mp4(input_path: Path, output_path: Path): #conver function, uses where the avi is stored and where we want to store the mp4
    cap = cv2.VideoCapture(str(input_path)) #opens video to read frame by frame, converts the path into a string
    if not cap.isOpened():
        raise IOError(f"Could not open input video: {input_path}")#if it cant open the video, returns a error

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0 #read fps if cant, set it to 24fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #get width of video
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #get height of video

    fourcc = cv2.VideoWriter_fourcc(*"avc1") #avc1 = h.264 codec, most compatiable with mp4 encoding, vidwriter turns it into a usable id
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height)) #tells opencv where to save mp4, which codec, video fps, resolution, out= object that writes each frame

    while True: #read->write loop, true if it reds frame
        ret, frame = cap.read()
        if not ret:#if it's false, no more frames, exit loop
            break 
        out.write(frame)#otherwise write the frame into mp4 video

    cap.release()#closes and finaliases video file
    out.release()#..^
    print(f"[INFO] saved mp4 to {output_path}")#shows where it's saved
