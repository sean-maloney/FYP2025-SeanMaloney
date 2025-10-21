# %%
# ✅ Install required dependencies (run once per environment)
%pip install ultralytics opencv-python pandas matplotlib tqdm


# %%
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")


# %%
import os
import cv2
import shutil
import random
import zipfile
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt


# %%
# ✅ Download VisDrone dataset from Ultralytics GitHub
from ultralytics.utils.downloads import download

dataset_dir = Path("datasets/VisDrone")
dataset_dir.mkdir(parents=True, exist_ok=True)

urls = [
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip",
]

download(urls, dir=dataset_dir, threads=4)

print("✅ VisDrone dataset downloaded and ready.")


# %%
# --- Convert the produced AVI to MP4 using 'avc1' ---
import cv2, glob

# Try to find the AVI in the detect subfolder we just used
avi_candidates = sorted(glob.glob(str(RUNS / "detect" / OUT_NAME / "*.avi")))
if not avi_candidates:
    raise FileNotFoundError("No .avi found in the detect output. Check the folder.")

input_avi = avi_candidates[0]
output_mp4 = str((RUNS / "detect" / OUT_NAME / "output_fixed.mp4").resolve())

cap = cv2.VideoCapture(input_avi)
if not cap.isOpened():
    raise IOError(f"Could not open input video: {input_avi}")

fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"avc1")
out = cv2.VideoWriter(output_mp4, fourcc, fps, (w, h))

n = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    n += 1

cap.release()
out.release()
print(f"✅ MP4 saved: {output_mp4}  | frames: {n} | fps: {fps:.2f} | size: {w}x{h}")


# %%
yaml_path = dataset_dir / "data.yaml"

yaml_path.write_text(f"""
path: {dataset_dir}

train: images/train
val: images/val
test: images/test

names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
""")

print("✅ data.yaml created at:", yaml_path)


# %%
import os
import shutil

folders_to_clean = ['runs/detect', 'runs/train']

for folder in folders_to_clean:
    if os.path.exists(folder):
        try:
            shutil.rmtree(folder, ignore_errors=False)
            print(f"🧹 Deleted old folder: {folder}")
        except PermissionError:
            print(f"⚠️ Permission denied: {folder}. Trying again safely...")
            # Try again with ignore_errors=True
            shutil.rmtree(folder, ignore_errors=True)
            print(f"✅ Force deleted (ignore_errors): {folder}")

print("✨ Workspace cleaned successfully.")


# %%
# ✅ Train YOLOv8 on VisDrone (detects vehicles and people)
model = YOLO("yolov8n.pt")
model.train(
    data=str(yaml_path),
    epochs=10,
    imgsz=640,
    batch=8,
    name="visdrone_parking_detector"
)


# %%
# ✅ Evaluate model performance on validation set
model.val(data=str(yaml_path))


# %%
# ✅ Run inference on your parking lot video
video_path = "Assets/Video1.mp4"
output_dir = "runs/detect/parking_output"

results = model.predict(source=video_path, save=True, project="runs/detect", name="parking_output", show=False)
print("🎥 Inference complete. Output saved to:", results[0].save_dir)


# %%
base = "datasets/VisDrone"
val_images = os.listdir(f"{base}/images/val")
random.shuffle(val_images)
num_test = int(0.1 * len(val_images))
test_images = val_images[:num_test]

os.makedirs(f"{base}/images/test", exist_ok=True)
os.makedirs(f"{base}/labels/test", exist_ok=True)

for img in test_images:
    name = os.path.splitext(img)[0]
    shutil.move(f"{base}/images/val/{img}", f"{base}/images/test/{img}")
    if os.path.exists(f"{base}/labels/val/{name}.txt"):
        shutil.move(f"{base}/labels/val/{name}.txt", f"{base}/labels/test/{name}.txt")

print(f"✅ Created test set with {len(test_images)} images and labels.")


# %%
for split in ["train", "val", "test"]:
    img_count = len(os.listdir(f"{dataset_dir}/images/{split}"))
    label_count = len(os.listdir(f"{dataset_dir}/labels/{split}"))
    print(f"{split.capitalize()} → Images: {img_count} | Labels: {label_count}")


# %%
import cv2
import os

# Paths
input_path = "runs/detect/parking_output/Video1.avi"   # change this if your AVI has a different name
output_path = "runs/detect/predict/output_fixed.mp4"

# Check input exists
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Input file not found: {input_path}")

# Open video
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("❌ Could not open input video.")

# Get properties
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video Info: {fps:.2f} FPS, {width}x{height}")

# Create MP4 writer
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Better compatibility than mp4v
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Convert frame by frame
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    frame_count += 1

# Clean up
cap.release()
out.release()

print(f"✅ Conversion complete! {frame_count} frames saved to: {output_path}")


USE THIS CODE, AND GET RID OF pedestrian, awning tricycle, tricycle and no people and no tricycle, DO NOT BREAK IT THIS TIME AND GIVE IT BACK TO ME IN CODE BLOCKS SO I CAN PUT IT INTO A JUPYTRE NOTEBOOK





