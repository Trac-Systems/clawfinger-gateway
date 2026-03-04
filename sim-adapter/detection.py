"""Object detection via YOLO-World — open-vocabulary detection on camera frames.

Used by the sim adapter (RTX) and future Jetson deployment.
Runs detection on JPEG frames given text prompts (e.g. "keys", "person", "cup").
Returns bounding boxes, confidence scores, and cropped images.
"""

from __future__ import annotations

import io
import time
from typing import Any

_model = None
_model_size: str = ""


def _ensure_model(size: str = "yolov8s-worldv2") -> Any:
    """Load YOLO-World model (cached)."""
    global _model, _model_size
    if _model is not None and _model_size == size:
        return _model
    try:
        from ultralytics import YOLO
        _model = YOLO(size)
        _model_size = size
        print(f"[detection] YOLO-World model loaded: {size}", flush=True)
        return _model
    except ImportError:
        print("[detection] ERROR: ultralytics not installed. pip install ultralytics", flush=True)
        return None
    except Exception as exc:
        print(f"[detection] ERROR loading YOLO-World: {exc}", flush=True)
        return None


def detect(frame_jpeg: bytes, classes: list[str],
           confidence: float = 0.3, max_detections: int = 10,
           model_size: str = "yolov8s-worldv2") -> list[dict]:
    """Run open-vocabulary detection on a JPEG frame.

    Args:
        frame_jpeg: JPEG-encoded image bytes
        classes: List of class names to detect (e.g. ["keys", "person", "cup"])
        confidence: Minimum confidence threshold (0-1)
        max_detections: Maximum number of detections to return
        model_size: YOLO-World model variant

    Returns:
        List of detection dicts:
        [{"label": str, "confidence": float, "bbox": [x1,y1,x2,y2],
          "cropped_b64": str, "center_x": float, "center_y": float}]
    """
    import base64

    model = _ensure_model(model_size)
    if model is None:
        return []

    try:
        from PIL import Image
        img = Image.open(io.BytesIO(frame_jpeg)).convert("RGB")
    except Exception as exc:
        print(f"[detection] Failed to decode image: {exc}", flush=True)
        return []

    t0 = time.time()

    # Set classes for open-vocabulary detection
    model.set_classes(classes)

    # Run inference
    results = model.predict(img, conf=confidence, verbose=False)

    detections = []
    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for i in range(min(len(boxes), max_detections)):
                box = boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_idx = int(box.cls[0])
                label = classes[cls_idx] if cls_idx < len(classes) else f"class_{cls_idx}"

                # Crop detection region
                cropped = img.crop((int(x1), int(y1), int(x2), int(y2)))
                crop_buf = io.BytesIO()
                cropped.save(crop_buf, format="JPEG", quality=70)
                crop_b64 = base64.b64encode(crop_buf.getvalue()).decode()

                detections.append({
                    "label": label,
                    "confidence": round(conf, 3),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "cropped_b64": crop_b64,
                    "center_x": round((x1 + x2) / 2, 1),
                    "center_y": round((y1 + y2) / 2, 1),
                })

    elapsed_ms = (time.time() - t0) * 1000
    if detections:
        print(f"[detection] Found {len(detections)} objects in {elapsed_ms:.0f}ms: "
              f"{', '.join(d['label'] for d in detections)}", flush=True)

    return detections


def check_available() -> dict:
    """Check if YOLO-World is available."""
    try:
        from ultralytics import YOLO
        return {"available": True, "package": "ultralytics"}
    except ImportError:
        return {"available": False, "error": "ultralytics not installed"}
