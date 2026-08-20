"""Standalone inference script for NVIDIA Jetson Nano 2GB.

Designed to run on the Jetson Nano 2GB with minimal dependencies.
Supports ONNX and TensorRT models, image files, and live camera input.

Usage (on Jetson Nano)
----------------------
    # Single image
    python3 jetson_inference.py --model pest_detector.onnx --source field_photo.jpg

    # USB Camera (device 0)
    python3 jetson_inference.py --model pest_detector.onnx --source 0

    # CSI Camera (Jetson Nano)
    python3 jetson_inference.py --model pest_detector.onnx --source csi

    # Batch inference on a folder
    python3 jetson_inference.py --model pest_detector.onnx --source images_folder/
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import sys

# Class names for this model (must match training order -- 8 rice classes)
CLASS_NAMES = {
    0: "rice_disease_bacterial_leaf_blight",
    1: "rice_disease_brown_spot",
    2: "rice_disease_false_smut",
    3: "rice_disease_leaf_sheath_blight",
    4: "rice_pest_leaf_folder",
    5: "rice_pest_rice_skipper",
    6: "rice_pest_white_stem_borer",
    7: "rice_pest_yellow_stem_borer",
}

# Colours for bounding boxes (BGR format)
BOX_COLOURS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 255, 0), (0, 128, 255),
]


def load_onnx_model(model_path: str):
    """Load ONNX model using onnxruntime."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "onnxruntime not installed. On Jetson Nano, install with:\n"
            "  pip3 install onnxruntime-gpu  # for GPU acceleration\n"
            "  pip3 install onnxruntime      # for CPU only"
        )

    # Prefer GPU (CUDA) execution provider on Jetson Nano
    providers = []
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if "TensorrtExecutionProvider" in available:
        providers.insert(0, "TensorrtExecutionProvider")
    providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(model_path, providers=providers)

    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape  # e.g. [1, 3, 416, 416]

    print(f"[+] ONNX model loaded: {model_path}")
    print(f"    Input name  : {input_name}")
    print(f"    Input shape : {input_shape}")
    print(f"    Provider    : {session.get_providers()[0]}")

    return session, input_name, input_shape


def preprocess(image: np.ndarray, input_size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Preprocess image for YOLO inference.
    Returns (blob, scale, original_size).
    """
    h, w = image.shape[:2]
    scale = min(input_size / h, input_size / w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    # Pad to square
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    # BGR -> RGB, HWC -> CHW, normalize to [0, 1]
    blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)  # Add batch dimension

    return blob, scale, (w, h)


def postprocess(
    output: np.ndarray,
    scale: float,
    original_size: tuple[int, int],
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
) -> list[dict]:
    """Post-process YOLO output to get detections.

    Handles both YOLO output formats:
    - [1, num_classes+4, num_detections] (Ultralytics default)
    - [1, num_detections, num_classes+4]
    """
    if output.ndim == 3:
        # Determine format
        if output.shape[1] < output.shape[2]:
            # Shape [1, num_classes+4, num_detections] -> transpose
            output = output.transpose(0, 2, 1)
        predictions = output[0]  # [num_detections, num_classes+4]
    else:
        predictions = output

    orig_w, orig_h = original_size
    detections = []

    for pred in predictions:
        # First 4 values are box coords (cx, cy, w, h)
        cx, cy, bw, bh = pred[:4]
        class_scores = pred[4:]

        # Get best class
        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])

        if confidence < conf_thresh:
            continue

        # Convert from centre format to corner format
        x1 = (cx - bw / 2) / scale
        y1 = (cy - bh / 2) / scale
        x2 = (cx + bw / 2) / scale
        y2 = (cy + bh / 2) / scale

        # Clamp to image bounds
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))

        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        detections.append({
            "box": [float(x1), float(y1), float(x2), float(y2)],
            "confidence": confidence,
            "class_id": class_id,
            "class_name": CLASS_NAMES.get(class_id, f"class_{class_id}"),
        })

    # NMS
    if not detections:
        return []

    boxes = np.array([d["box"] for d in detections])
    scores = np.array([d["confidence"] for d in detections])
    class_ids = np.array([d["class_id"] for d in detections])

    keep = []
    for cls in set(class_ids):
        cls_mask = class_ids == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_indices = np.where(cls_mask)[0]

        order = cls_scores.argsort()[::-1]
        suppressed = set()

        for i in range(len(order)):
            idx_i = order[i]
            if idx_i in suppressed:
                continue
            keep.append(cls_indices[idx_i])

            for j in range(i + 1, len(order)):
                idx_j = order[j]
                if idx_j in suppressed:
                    continue

                # Compute IoU
                b1, b2 = cls_boxes[idx_i], cls_boxes[idx_j]
                xi1 = max(b1[0], b2[0])
                yi1 = max(b1[1], b2[1])
                xi2 = min(b1[2], b2[2])
                yi2 = min(b1[3], b2[3])
                inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                iou = inter / max(1e-6, area1 + area2 - inter)

                if iou > iou_thresh:
                    suppressed.add(idx_j)

    return [detections[i] for i in keep]


def draw_detections(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes and labels on the image."""
    annotated = image.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        cls_id = det["class_id"]
        conf = det["confidence"]
        name = det["class_name"]
        colour = BOX_COLOURS[cls_id % len(BOX_COLOURS)]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

        label = f"{name} {conf:.2f}"
        font_scale = 0.5
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw, y1), colour, -1)
        cv2.putText(annotated, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    return annotated


def get_csi_pipeline(
    sensor_id: int = 0,
    capture_width: int = 1280,
    capture_height: int = 720,
    display_width: int = 640,
    display_height: int = 480,
    framerate: int = 30,
    flip_method: int = 0,
) -> str:
    """GStreamer pipeline for Jetson Nano CSI camera."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, "
        f"height=(int){capture_height}, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, "
        f"format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )


def run_image(
    session, input_name: str, input_size: int,
    image_path: str, conf: float, iou: float, output_dir: Path,
):
    """Run inference on a single image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"[!] Cannot read image: {image_path}")
        return

    blob, scale, orig_size = preprocess(image, input_size)
    start = time.time()
    outputs = session.run(None, {input_name: blob})
    latency = (time.time() - start) * 1000

    detections = postprocess(outputs[0], scale, orig_size, conf, iou)
    annotated = draw_detections(image, detections)

    output_path = output_dir / f"{Path(image_path).stem}_detections.jpg"
    cv2.imwrite(str(output_path), annotated)

    print(f"  {Path(image_path).name}: {len(detections)} detections, {latency:.1f}ms")
    for det in detections:
        print(f"    - {det['class_name']} ({det['confidence']:.2f})")

    return detections


def run_camera(
    session, input_name: str, input_size: int,
    source: str, conf: float, iou: float,
):
    """Run live inference from camera."""
    if source == "csi":
        cap = cv2.VideoCapture(get_csi_pipeline(), cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(int(source))

    if not cap.isOpened():
        print(f"[!] Cannot open camera: {source}")
        return

    print(f"[+] Camera opened. Press 'q' to quit, 's' to save screenshot.")

    fps_history = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        blob, scale, orig_size = preprocess(frame, input_size)
        start = time.time()
        outputs = session.run(None, {input_name: blob})
        latency = (time.time() - start) * 1000

        detections = postprocess(outputs[0], scale, orig_size, conf, iou)
        annotated = draw_detections(frame, detections)

        # FPS display
        fps = 1000 / max(latency, 1)
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        cv2.putText(
            annotated,
            f"FPS: {avg_fps:.1f} | Detections: {len(detections)}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )

        cv2.imshow("Pest Detection - Jetson Nano", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            save_path = f"screenshot_{frame_count}.jpg"
            cv2.imwrite(save_path, annotated)
            print(f"[+] Screenshot saved: {save_path}")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"[+] Camera closed after {frame_count} frames.")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO Pest Detection — Jetson Nano Inference"
    )
    parser.add_argument("--model", required=True,
                        help="Path to ONNX or TensorRT model")
    parser.add_argument("--source", required=True,
                        help="Image path, folder path, camera ID (0), or 'csi'")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--imgsz", type=int, default=320,
                        help="Input image size (must match export, 320 for Jetson Nano 2GB)")
    parser.add_argument("--output", type=Path, default=Path("detections"),
                        help="Output directory for annotated images")
    parser.add_argument("--classes", type=Path, default=None,
                        help="Optional JSON file mapping class IDs to names")
    args = parser.parse_args()

    # Load custom class names if provided
    global CLASS_NAMES
    if args.classes and args.classes.is_file():
        data = json.loads(args.classes.read_text(encoding="utf-8"))
        if "id_to_name" in data:
            CLASS_NAMES = {int(k): v for k, v in data["id_to_name"].items()}
        elif "names" in data:
            CLASS_NAMES = {int(k): v for k, v in data["names"].items()}

    # Load model
    session, input_name, input_shape = load_onnx_model(args.model)
    input_size = args.imgsz

    # Determine source type
    source = args.source.strip()

    if source.isdigit() or source == "csi":
        # Camera mode
        run_camera(session, input_name, input_size, source, args.conf, args.iou)
    elif Path(source).is_dir():
        # Batch folder mode
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = sorted(
            p for p in Path(source).iterdir()
            if p.suffix.lower() in image_exts
        )
        args.output.mkdir(parents=True, exist_ok=True)
        print(f"[+] Processing {len(image_files)} images from {source}")
        all_dets = []
        for img_path in image_files:
            dets = run_image(
                session, input_name, input_size,
                str(img_path), args.conf, args.iou, args.output,
            )
            if dets:
                all_dets.extend(dets)
        print(f"\n[+] Total detections: {len(all_dets)}")
        print(f"[+] Results saved to: {args.output}")
    elif Path(source).is_file():
        # Single image mode
        args.output.mkdir(parents=True, exist_ok=True)
        run_image(
            session, input_name, input_size,
            source, args.conf, args.iou, args.output,
        )
    else:
        print(f"[!] Source not found: {source}")
        sys.exit(1)


if __name__ == "__main__":
    main()
