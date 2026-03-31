from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from .config import YOLO_WEIGHTS_PATH, YOLO_DEVICE

yolo_model: Optional[YOLO] = None


def load_yolo_model() -> None:
    global yolo_model

    if yolo_model is not None:
        return

    if not YOLO_WEIGHTS_PATH.exists():
        yolo_model = None
        return

    model = YOLO(str(YOLO_WEIGHTS_PATH))
    try:
        model.to(YOLO_DEVICE)
    except Exception:
        pass
    yolo_model = model


def get_yolo_model() -> YOLO:
    global yolo_model
    if yolo_model is None:
        load_yolo_model()
    if yolo_model is None:
        raise RuntimeError(f"YOLO weights not found at {YOLO_WEIGHTS_PATH}")
    return yolo_model


def _norm_poly_to_px(poly: List[Dict[str, float]], w: int, h: int) -> np.ndarray:
    pts = [(int(p["x"] * w), int(p["y"] * h)) for p in poly]
    return np.array(pts, dtype=np.int32).reshape((-1, 1, 2))


def _is_point_in_poly(px: int, py: int, poly_np: np.ndarray) -> bool:
    return cv2.pointPolygonTest(poly_np, (float(px), float(py)), False) >= 0


def _bottom_center(x1: float, y1: float, x2: float, y2: float) -> Tuple[int, int]:
    return int((x1 + x2) / 2.0), int(y2)


def analyze_image_with_spots(
    input_image: Path,
    spots_doc: Dict[str, Any],
    conf: float = 0.25,
    vehicle_class_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    image = cv2.imread(str(input_image))
    if image is None:
        raise RuntimeError(f"Could not read image: {input_image}")

    model = get_yolo_model()
    result = model.predict(image, conf=conf, verbose=False, device=YOLO_DEVICE)[0]

    points: List[Tuple[int, int]] = []
    if result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else None
        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            if vehicle_class_ids is not None and cls is not None and cls[i] not in vehicle_class_ids:
                continue
            points.append(_bottom_center(x1, y1, x2, y2))

    available = 0
    occupied = 0
    spots = []

    h, w = image.shape[:2]

    for s in spots_doc.get("spots", []):
        sid = s.get("id", "")
        stype = s.get("type", "parking")
        description = s.get("description", "")
        poly = s.get("polygon", [])
        if not sid or len(poly) < 3:
            continue

        poly_np = _norm_poly_to_px(poly, w, h)
        is_occupied = False

        for px, py in points:
            if _is_point_in_poly(px, py, poly_np):
                is_occupied = True
                break

        status = "occupied" if is_occupied else "available"
        spots.append({
            "id": sid,
            "type": stype,
            "description": description,
            "status": status,
        })

        if is_occupied:
            occupied += 1
        else:
            available += 1

    return {
        "available": available,
        "occupied": occupied,
        "spots": spots,
    }


def run_inference_with_spots(
    input_video: Path,
    output_dir: Path,
    spots_doc: Dict[str, Any],
    conf: float = 0.25,
    vehicle_class_ids: Optional[List[int]] = None,
) -> Tuple[Path, int, int]:
    if yolo_model is None:
        raise RuntimeError("YOLO model not loaded")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "annotated.mp4"

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not out.isOpened():
        cap.release()
        raise RuntimeError("Could not open VideoWriter for output video")

    spots_poly = []
    for s in spots_doc.get("spots", []):
        sid = s.get("id", "")
        stype = s.get("type", "parking")
        description = s.get("description", "")
        poly = s.get("polygon", [])
        if sid and len(poly) >= 3:
            spots_poly.append((sid, stype, description, _norm_poly_to_px(poly, w, h)))

    last_available = 0
    last_occupied = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        r = yolo_model.predict(frame, conf=conf, verbose=False, device=YOLO_DEVICE)[0]
        points = []

        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int) if r.boxes.cls is not None else None
            for i, (x1, y1, x2, y2) in enumerate(xyxy):
                if vehicle_class_ids is not None and cls is not None and cls[i] not in vehicle_class_ids:
                    continue
                points.append(_bottom_center(x1, y1, x2, y2))

        occupied_count = 0
        available_count = 0

        for sid, stype, description, poly_np in spots_poly:
            occupied = False
            for px, py in points:
                if _is_point_in_poly(px, py, poly_np):
                    occupied = True
                    break

            color = (0, 0, 255) if occupied else (0, 255, 0)
            cv2.polylines(frame, [poly_np], True, color, 2)
            x0, y0 = int(poly_np[0][0][0]), int(poly_np[0][0][1])
            label = sid if not stype else f"{sid} ({stype})"
            cv2.putText(frame, label, (x0 + 6, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            if occupied:
                occupied_count += 1
            else:
                available_count += 1

        last_available = available_count
        last_occupied = occupied_count

        cv2.putText(
            frame,
            f"Available: {available_count}  Occupied: {occupied_count}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        out.write(frame)

    cap.release()
    out.release()

    return out_path, last_available, last_occupied