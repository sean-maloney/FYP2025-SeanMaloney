from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from .config import YOLO_WEIGHTS_PATH, YOLO_DEVICE

yolo_model: YOLO | None = None


def load_yolo_model():
    global yolo_model

    if not YOLO_WEIGHTS_PATH.exists():
        yolo_model = None
        return

    model = YOLO(str(YOLO_WEIGHTS_PATH))
    try:
        model.to(YOLO_DEVICE)
    except Exception:
        pass
    yolo_model = model


def _norm_poly_to_px(poly: List[Dict[str, float]], w: int, h: int) -> np.ndarray:
    pts = [(int(p["x"] * w), int(p["y"] * h)) for p in poly]
    return np.array(pts, dtype=np.int32).reshape((-1, 1, 2))


def _is_point_in_poly(px: int, py: int, poly_np: np.ndarray) -> bool:
    return cv2.pointPolygonTest(poly_np, (float(px), float(py)), False) >= 0


def _bottom_center(x1: float, y1: float, x2: float, y2: float) -> Tuple[int, int]:
    return int((x1 + x2) / 2.0), int(y2)


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

    out = None
    for fourcc_str in ("avc1", "H264", "X264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        if out.isOpened():
            break
    if out is None or not out.isOpened():
        raise RuntimeError("Could not open VideoWriter")

    spots_poly: List[Tuple[str, np.ndarray]] = []
    for s in spots_doc.get("spots", []):
        sid = s.get("id", "")
        poly = s.get("polygon", [])
        if sid and len(poly) >= 3:
            spots_poly.append((sid, _norm_poly_to_px(poly, w, h)))

    last_available = 0
    last_occupied = 0

    frame_i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        r = yolo_model.predict(frame, conf=conf, verbose=False)[0]

        points = []
        boxes = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int) if r.boxes.cls is not None else None

            for i, (x1, y1, x2, y2) in enumerate(xyxy):
                class_id = int(cls[i]) if cls is not None else None
                if vehicle_class_ids is not None and class_id is not None:
                    if class_id not in vehicle_class_ids:
                        continue
                if vehicle_class_ids is None and class_id is not None:
                    if class_id not in {2, 3, 5, 7}:
                        continue
                points.append(_bottom_center(x1, y1, x2, y2))
                boxes.append((int(x1), int(y1), int(x2), int(y2), class_id))

        for x1, y1, x2, y2, class_id in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            if class_id is not None and hasattr(r, "names") and isinstance(r.names, dict):
                name = r.names.get(class_id, str(class_id))
                cv2.putText(
                    frame,
                    name,
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        occupied_count = 0
        available_count = 0

        for sid, poly_np in spots_poly:
            occupied = False
            for px, py in points:
                if _is_point_in_poly(px, py, poly_np):
                    occupied = True
                    break

            color = (0, 0, 255) if occupied else (0, 255, 0)
            cv2.polylines(frame, [poly_np], True, color, 2)

            x0, y0 = int(poly_np[0][0][0]), int(poly_np[0][0][1])
            cv2.putText(
                frame,
                sid,
                (x0 + 6, y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

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

        frame_i += 1
        if frame_i % 30 == 0:
            print(f"inference frames={frame_i} available={available_count} occupied={occupied_count}")

    cap.release()
    out.release()

    return out_path, last_available, last_occupied
