"""test_model.py — Test and Deployment Readiness Checker for Rice Pest & Disease YOLO Model

Usage
-----
  # Test on a single image
  python test_model.py --image path/to/your/image.jpg

  # Test on a folder of images
  python test_model.py --image path/to/folder/

  # Test on webcam (live)
  python test_model.py --source 0

  # Use a specific weights file
  python test_model.py --image photo.jpg --weights runs/detect/runs/pest_disease_rice_yolo11s/weights/best.pt

  # Run only the deployment readiness check
  python test_model.py --check-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ─── Class Configuration ──────────────────────────────────────────────────
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

# Color per class (BGR): Diseases = warm, Pests = cool/green
CLASS_COLORS = {
    0: (0,  80, 220),   # Red-Orange  — Bacterial Blight
    1: (0,  55, 180),   # Dark Red     — Brown Spot
    2: (0, 200, 255),   # Gold/Yellow  — False Smut
    3: (130, 0, 200),   # Purple       — Sheath Blight
    4: (0, 210,  60),   # Bright Green — Leaf Folder
    5: (30, 180, 255),  # Amber        — Rice Skipper
    6: (255, 255, 80),  # Cyan-White   — White Stem Borer
    7: (0, 255, 200),   # Mint         — Yellow Stem Borer
}

# Default weights path (most recently trained)
DEFAULT_WEIGHTS = "runs/detect/runs/pest_disease_rice_yolo11s/weights/best.pt"


# ─── Deployment Readiness Check ───────────────────────────────────────────

def check_deployment_readiness(weights_path: Path) -> dict:
    """Run a comprehensive deployment readiness check and return results."""
    results = {
        "pass": [],
        "warn": [],
        "fail": [],
        "ready": True,
    }

    print("\n" + "=" * 62)
    print("  DEPLOYMENT READINESS CHECK")
    print("=" * 62)

    # 1. Check weights file exists
    if weights_path.exists():
        size_mb = weights_path.stat().st_size / 1e6
        results["pass"].append(f"Weights file exists ({size_mb:.1f} MB): {weights_path}")
        print(f"  [PASS] Weights found: {weights_path.name} ({size_mb:.1f} MB)")
    else:
        results["fail"].append(f"Weights file NOT found: {weights_path}")
        results["ready"] = False
        print(f"  [FAIL] Weights NOT found: {weights_path}")
        print("         Train first: python train_yolo11.py --device 0")
        return results

    # 2. Check PyTorch / CUDA
    try:
        import torch
        results["pass"].append(f"PyTorch {torch.__version__} installed")
        print(f"  [PASS] PyTorch {torch.__version__}")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            results["pass"].append(f"CUDA GPU: {gpu_name} ({vram:.1f} GB VRAM)")
            print(f"  [PASS] GPU: {gpu_name} ({vram:.1f} GB VRAM)")
        else:
            results["warn"].append("CUDA not available — will run on CPU (slow)")
            print("  [WARN] No CUDA GPU — inference will use CPU (slower)")
    except ImportError:
        results["fail"].append("PyTorch not installed")
        results["ready"] = False
        print("  [FAIL] PyTorch not installed: pip install torch")

    # 3. Check Ultralytics
    try:
        import ultralytics
        results["pass"].append(f"Ultralytics {ultralytics.__version__} installed")
        print(f"  [PASS] Ultralytics {ultralytics.__version__}")
    except ImportError:
        results["fail"].append("Ultralytics not installed")
        results["ready"] = False
        print("  [FAIL] Ultralytics not installed: pip install ultralytics")

    # 4. Check OpenCV
    try:
        results["pass"].append(f"OpenCV {cv2.__version__} installed")
        print(f"  [PASS] OpenCV {cv2.__version__}")
    except Exception:
        results["fail"].append("OpenCV not available")
        results["ready"] = False
        print("  [FAIL] OpenCV not available")

    # 5. Load model and check class count
    try:
        from ultralytics import YOLO
        print("  [INFO] Loading model weights...")
        model = YOLO(str(weights_path))
        nc = model.model.nc
        if nc == len(CLASS_NAMES):
            results["pass"].append(f"Model has correct {nc} classes")
            print(f"  [PASS] Model classes: {nc} (matches 8-class config)")
        else:
            results["fail"].append(f"Class mismatch: model has {nc}, expected {len(CLASS_NAMES)}")
            results["ready"] = False
            print(f"  [FAIL] Class mismatch! Model: {nc}, expected: {len(CLASS_NAMES)}")
    except Exception as e:
        results["fail"].append(f"Model load error: {e}")
        results["ready"] = False
        print(f"  [FAIL] Model load error: {e}")
        return results

    # 6. Dummy inference speed test
    try:
        dummy_img = np.random.randint(0, 255, (480, 480, 3), dtype=np.uint8)
        # Warmup
        model.predict(dummy_img, verbose=False)
        # Timed run
        t0 = time.perf_counter()
        for _ in range(5):
            model.predict(dummy_img, verbose=False)
        elapsed = (time.perf_counter() - t0) / 5
        fps = 1.0 / elapsed
        print(f"  [INFO] Inference speed: {elapsed*1000:.1f} ms/image ({fps:.1f} FPS)")

        if fps >= 5:
            results["pass"].append(f"Inference speed: {fps:.1f} FPS ({elapsed*1000:.1f} ms)")
            print(f"  [PASS] Speed OK for deployment ({fps:.1f} FPS)")
        else:
            results["warn"].append(f"Inference slow: {fps:.1f} FPS — consider smaller imgsz or TensorRT export")
            print(f"  [WARN] Inference slow ({fps:.1f} FPS) — consider export_jetson.py for Jetson")
    except Exception as e:
        results["warn"].append(f"Speed test failed: {e}")
        print(f"  [WARN] Speed test failed: {e}")

    # 7. Check ONNX export capability
    try:
        import onnx
        results["pass"].append(f"ONNX {onnx.__version__} available (Jetson export ready)")
        print(f"  [PASS] ONNX {onnx.__version__} (Jetson ONNX export supported)")
    except ImportError:
        results["warn"].append("ONNX not installed — run: pip install onnx onnxruntime")
        print("  [WARN] ONNX not installed (needed for Jetson export)")

    # 8. Summary
    print("\n" + "-" * 62)
    print(f"  PASSED  : {len(results['pass'])}")
    print(f"  WARNINGS: {len(results['warn'])}")
    print(f"  FAILED  : {len(results['fail'])}")
    print("-" * 62)

    if results["fail"]:
        results["ready"] = False
        print("  STATUS  : NOT READY FOR DEPLOYMENT")
        for f in results["fail"]:
            print(f"            -> {f}")
    elif results["warn"]:
        print("  STATUS  : READY WITH WARNINGS (review above)")
    else:
        print("  STATUS  : FULLY READY FOR DEPLOYMENT")

    print("=" * 62)
    return results


# ─── Single Image Inference ───────────────────────────────────────────────

def draw_detections(img: np.ndarray, boxes, conf_threshold: float = 0.25) -> tuple[np.ndarray, list]:
    """Draw bounding boxes and labels on image. Returns annotated image and detections list."""
    annotated = img.copy()
    h, w = annotated.shape[:2]
    detections = []

    for box in boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue

        cid = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color = CLASS_COLORS.get(cid, (0, 255, 0))
        name = CLASS_NAMES.get(cid, f"class_{cid}")
        label = f"{name.split('_', 2)[-1].replace('_', ' ')}: {conf:.2f}"

        # Draw filled box with transparency
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.12, annotated, 0.88, 0, annotated)

        # Draw border
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label badge
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(0.7, w / 900))
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        badge_y1 = max(0, y1 - th - 10)
        badge_y2 = y1
        cv2.rectangle(annotated, (x1, badge_y1), (x1 + tw + 10, badge_y2), color, -1)
        cv2.putText(annotated, label, (x1 + 5, badge_y2 - 5), font, font_scale, (255, 255, 255), thickness)

        detections.append({
            "class_id": cid,
            "class_name": name,
            "confidence": round(conf, 4),
            "bbox_xyxy": [x1, y1, x2, y2],
        })

    return annotated, detections


def run_inference(model, img_path: Path, conf: float, imgsz: int, save_dir: Path) -> list:
    """Run inference on a single image file."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [ERROR] Could not read image: {img_path}")
        return []

    h, w = img.shape[:2]
    print(f"\n  Image : {img_path.name}  ({w}x{h} px)")

    t0 = time.perf_counter()
    results = model.predict(img, conf=conf, imgsz=imgsz, verbose=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"  Time  : {elapsed_ms:.1f} ms")

    boxes = results[0].boxes if results[0].boxes else []
    annotated, detections = draw_detections(img, boxes, conf)

    if detections:
        print(f"  Found : {len(detections)} detection(s)")
        for d in detections:
            category = "DISEASE" if "disease" in d["class_name"] else "PEST"
            short_name = d["class_name"].split("_", 2)[-1].replace("_", " ").title()
            print(f"          [{category}] {short_name}  conf={d['confidence']:.3f}  box={d['bbox_xyxy']}")
    else:
        print(f"  Found : No detections (conf threshold={conf})")
        # Optionally lower threshold suggestion
        low_results = model.predict(img, conf=0.1, imgsz=imgsz, verbose=False)
        low_boxes = low_results[0].boxes if low_results[0].boxes else []
        if low_boxes:
            print(f"  Hint  : {len(low_boxes)} detection(s) found at conf=0.10 — try --conf 0.15")

    # Save annotated image
    out_path = save_dir / f"detected_{img_path.stem}{img_path.suffix}"
    cv2.imwrite(str(out_path), annotated)
    print(f"  Saved : {out_path}")

    # Save JSON report
    json_path = save_dir / f"detected_{img_path.stem}.json"
    report = {
        "image": str(img_path),
        "width": w,
        "height": h,
        "inference_ms": round(elapsed_ms, 2),
        "conf_threshold": conf,
        "detections": detections,
        "summary": {
            "total": len(detections),
            "diseases": sum(1 for d in detections if "disease" in d["class_name"]),
            "pests": sum(1 for d in detections if "pest" in d["class_name"]),
        }
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  JSON  : {json_path}")

    return detections


def run_live_camera(model, source: int, conf: float, imgsz: int):
    """Run real-time inference on webcam or camera index."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera source: {source}")
        return

    print(f"\n[INFO] Live camera inference (source={source}). Press Q to quit.")
    fps_avg = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        results = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)
        elapsed = time.perf_counter() - t0
        fps_avg.append(1.0 / elapsed)
        if len(fps_avg) > 30:
            fps_avg.pop(0)

        boxes = results[0].boxes if results[0].boxes else []
        annotated, _ = draw_detections(frame, boxes, conf)

        # FPS overlay
        fps_text = f"FPS: {sum(fps_avg)/len(fps_avg):.1f}"
        cv2.putText(annotated, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 100), 2)

        cv2.imshow("Rice Pest & Disease Detector (Q=quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test and Deployment Readiness Checker for Rice Pest Detection Model"
    )
    parser.add_argument(
        "--weights", type=Path, default=Path(DEFAULT_WEIGHTS),
        help=f"Path to YOLO .pt weights file (default: {DEFAULT_WEIGHTS})"
    )
    parser.add_argument(
        "--image", type=Path, default=None,
        help="Path to an image file or folder of images to test"
    )
    parser.add_argument(
        "--source", type=int, default=None,
        help="Camera index for live inference (e.g. 0 for webcam)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold 0-1 (default: 0.25)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=480,
        help="Inference image size (default: 480, use 320 for Jetson Nano)"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("test_results"),
        help="Directory to save annotated images and JSON reports (default: test_results/)"
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Only run deployment readiness check, skip image inference"
    )
    parser.add_argument(
        "--no-check", action="store_true",
        help="Skip deployment readiness check, go straight to inference"
    )
    args = parser.parse_args()

    print("\n" + "=" * 62)
    print("  Rice Pest & Disease Detection — Test & Readiness Tool")
    print("=" * 62)
    print(f"  Weights : {args.weights}")
    print(f"  Conf    : {args.conf}")
    print(f"  Imgsz   : {args.imgsz}")

    # ── Step 1: Deployment readiness check ──
    if not args.no_check:
        check_results = check_deployment_readiness(args.weights)
        if args.check_only:
            sys.exit(0 if check_results["ready"] else 1)
        if not check_results["ready"]:
            print("\n[ABORT] Deployment check failed — fix issues above before testing.")
            sys.exit(1)

    # ── Step 2: Load model ──
    try:
        from ultralytics import YOLO
        print(f"\n[INFO] Loading model: {args.weights}")
        model = YOLO(str(args.weights))
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    # ── Step 3: Run inference ──
    if args.source is not None:
        run_live_camera(model, args.source, args.conf, args.imgsz)

    elif args.image is not None:
        if args.image.is_dir():
            # Batch: all images in folder
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            images = [f for f in args.image.iterdir() if f.suffix.lower() in exts]
            if not images:
                print(f"[ERROR] No images found in {args.image}")
                sys.exit(1)
            print(f"\n[INFO] Found {len(images)} images in {args.image}")
            all_detections = []
            for img_path in sorted(images):
                dets = run_inference(model, img_path, args.conf, args.imgsz, args.output)
                all_detections.extend(dets)

            # Summary report
            print(f"\n{'=' * 62}")
            print(f"  BATCH SUMMARY: {len(images)} images")
            from collections import Counter
            class_counts = Counter(d["class_name"] for d in all_detections)
            for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
                tag = "DISEASE" if "disease" in cls else "PEST"
                print(f"  [{tag}] {cls}: {cnt} detection(s)")
            if not all_detections:
                print("  No detections found across all images.")
            print(f"{'=' * 62}\n")

        elif args.image.is_file():
            run_inference(model, args.image, args.conf, args.imgsz, args.output)
        else:
            print(f"[ERROR] Path not found: {args.image}")
            sys.exit(1)

    else:
        print("\n[INFO] No --image or --source specified.")
        print("       Deployment readiness check completed above.")
        print("\nUsage examples:")
        print("  python test_model.py --image my_rice_photo.jpg")
        print("  python test_model.py --image my_photos_folder/")
        print("  python test_model.py --source 0   # webcam")
        print("  python test_model.py --check-only  # just check readiness")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
