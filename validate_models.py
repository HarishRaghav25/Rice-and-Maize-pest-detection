"""Comprehensive validation and comparison of YOLOv11s and YOLOv26s models.

Runs validation on both models, reports all metrics, and generates a
comparison table. Use this after training to verify both models meet
the 90% mAP@50 target.

Usage
-----
    python validate_models.py --data yolo_dataset_v2/dataset.yaml
    python validate_models.py --yolo11 runs/.../best.pt --yolo26 runs/.../best.pt --data yolo_dataset_v2/dataset.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


TARGET_MAP50 = 0.90  # 90% mAP@50 target


def find_best_weights(project: str, pattern: str) -> Path | None:
    """Search for the best.pt weights in recent runs."""
    runs_dir = Path(project) / "detect"
    if not runs_dir.exists():
        runs_dir = Path(project)
    if not runs_dir.exists():
        return None

    candidates = sorted(runs_dir.glob(f"*{pattern}*/weights/best.pt"), reverse=True)
    return candidates[0] if candidates else None


def validate_model(
    model_path: str | Path,
    data_path: str,
    model_name: str,
    imgsz: int = 640,
    device: str | None = None,
) -> dict:
    """Validate a single model and return metrics dict."""
    print(f"\n{'=' * 60}")
    print(f"  Validating: {model_name}")
    print(f"  Weights   : {model_path}")
    print(f"{'=' * 60}")

    model = YOLO(str(model_path))
    results = model.val(
        data=data_path,
        imgsz=imgsz,
        device=device,
        split="val",
    )

    box = results.box
    map50 = float(box.map50)
    map50_95 = float(box.map)
    mp = float(box.mp)
    mr = float(box.mr)
    f1 = 2 * mp * mr / (mp + mr) if (mp + mr) > 0 else 0.0
    fitness = float(results.fitness)

    # Per-class metrics
    per_class = []
    if hasattr(box, "maps") and len(box.maps) > 0 and hasattr(results, "names"):
        p_class = getattr(box, "p", [])
        r_class = getattr(box, "r", [])

        for idx, map_val in enumerate(box.maps):
            cls_name = results.names.get(idx, f"class_{idx}")
            cls_map50 = float(box.class_result(idx)[2]) if hasattr(box, "class_result") else 0.0
            p_val = float(p_class[idx]) if idx < len(p_class) else 0.0
            r_val = float(r_class[idx]) if idx < len(r_class) else 0.0
            cls_f1 = 2 * p_val * r_val / (p_val + r_val) if (p_val + r_val) > 0 else 0.0

            per_class.append({
                "class_id": idx,
                "class_name": cls_name,
                "mAP50": round(cls_map50, 4),
                "mAP50_95": round(float(map_val), 4),
                "precision": round(p_val, 4),
                "recall": round(r_val, 4),
                "f1": round(cls_f1, 4),
            })

    summary = {
        "model": model_name,
        "weights": str(model_path),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "Precision": round(mp, 4),
        "Recall": round(mr, 4),
        "F1_Score": round(f1, 4),
        "Fitness": round(fitness, 4),
        "target_met": map50 >= TARGET_MAP50,
        "per_class": per_class,
    }

    # Print results
    status = "✅ PASSED" if map50 >= TARGET_MAP50 else "❌ BELOW TARGET"
    print(f"\n  mAP@50       : {map50:.4f} ({map50 * 100:.2f}%)")
    print(f"  mAP@50-95    : {map50_95:.4f} ({map50_95 * 100:.2f}%)")
    print(f"  Precision    : {mp:.4f} ({mp * 100:.2f}%)")
    print(f"  Recall       : {mr:.4f} ({mr * 100:.2f}%)")
    print(f"  F1-Score     : {f1:.4f} ({f1 * 100:.2f}%)")
    print(f"  Fitness      : {fitness:.4f}")
    print(f"  Target (≥90%): {status}")

    if per_class:
        print(f"\n  Per-Class Breakdown:")
        print(f"  {'ID':<5} {'Class Name':<40} {'mAP50':<10} {'P':<10} {'R':<10} {'F1':<10}")
        print(f"  {'-' * 85}")
        for c in per_class:
            print(f"  {c['class_id']:<5} {c['class_name']:<40} {c['mAP50']:<10.4f} {c['precision']:<10.4f} {c['recall']:<10.4f} {c['f1']:<10.4f}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Validate YOLOv11s and YOLOv26s models and check 90%+ mAP@50 target."
    )
    parser.add_argument("--data", type=Path, required=True,
                        help="Path to dataset.yaml")
    parser.add_argument("--yolo11", type=Path, default=None,
                        help="Path to YOLOv11 best.pt (auto-detected if not specified)")
    parser.add_argument("--yolo26", type=Path, default=None,
                        help="Path to YOLOv26 best.pt (auto-detected if not specified)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=Path("runs/validation_results.json"))
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"Dataset not found: {args.data}")

    data_str = str(args.data.resolve())
    results = {}

    # YOLOv11
    yolo11_path = args.yolo11
    if yolo11_path is None:
        yolo11_path = find_best_weights("runs", "yolo11")
    if yolo11_path and yolo11_path.is_file():
        results["yolo11s"] = validate_model(
            yolo11_path, data_str, "YOLOv11s", args.imgsz, args.device
        )
    else:
        print(f"\n[!] YOLOv11 weights not found. Skipping YOLOv11 validation.")

    # YOLOv26
    yolo26_path = args.yolo26
    if yolo26_path is None:
        yolo26_path = find_best_weights("runs", "yolo26")
    if yolo26_path and yolo26_path.is_file():
        results["yolo26s"] = validate_model(
            yolo26_path, data_str, "YOLOv26s", args.imgsz, args.device
        )
    else:
        print(f"\n[!] YOLOv26 weights not found. Skipping YOLOv26 validation.")

    # Print comparison table
    if len(results) >= 2:
        y11 = results["yolo11s"]
        y26 = results["yolo26s"]

        print(f"\n\n{'=' * 70}")
        print(f"  COMPARATIVE VALIDATION RESULTS")
        print(f"{'=' * 70}")
        print(f"  {'Metric':<20} {'YOLOv11s':<20} {'YOLOv26s':<20}")
        print(f"  {'-' * 60}")
        for key in ["mAP50", "mAP50_95", "Precision", "Recall", "F1_Score", "Fitness"]:
            v11 = y11.get(key, 0)
            v26 = y26.get(key, 0)
            print(f"  {key:<20} {v11:<20.4f} {v26:<20.4f}")

        print(f"\n  YOLOv11s target met: {'✅ YES' if y11['target_met'] else '❌ NO'}")
        print(f"  YOLOv26s target met: {'✅ YES' if y26['target_met'] else '❌ NO'}")
        print(f"{'=' * 70}")

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[+] Results saved to: {args.output}")


if __name__ == "__main__":
    main()
