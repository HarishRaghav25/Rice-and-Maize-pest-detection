"""Train a YOLO26 detector for ICAR rice and maize pest and disease detection.
This script initializes and trains a YOLO26 object detection model on GPU,
performs validation, and outputs evaluation scores including mAP@50, mAP@50-95,
Precision, Recall, F1-Score, and per-class metrics.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml
from ultralytics import YOLO
from ultralytics.utils.checks import check_yolo

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def slug(text: str) -> str:
    text = re.sub(r"^\d+[_ -]*", "", text.lower())
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def class_for(image: Path, source: Path) -> str | None:
    relative = image.relative_to(source)
    parts = relative.parts
    crop = slug(parts[0])

    if len(parts) == 3 and parts[1].lower() == "healthy":
        return None
    if len(parts) < 4 or parts[1].lower() not in {"disease", "insect-pests"}:
        return None
    kind = "pest" if parts[1].lower() == "insect-pests" else "disease"
    return f"{crop}_{kind}_{slug(parts[2])}"


def generate_pseudo_yolo_dataset(source: Path, output: Path, seed: int = 42) -> Path:
    """Generate object detection dataset using full-frame bounding boxes from raw folder structure."""
    print(f"\n[+] Automatically building object detection dataset at '{output}' from '{source}'...")
    images = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)

    if not images:
        sys.exit(f"Error: No images found under '{source}'")

    classes = sorted({c for p in images if (c := class_for(p, source))})
    name_to_id = {name: i for i, name in enumerate(classes)}

    records = [(image, class_for(image, source)) for image in images]
    
    # Stratified split
    groups: dict[str, list[tuple[Path, str | None]]] = {}
    for record in records:
        groups.setdefault(record[1] or "__negative__", []).append(record)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}

    for group in groups.values():
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, round(n * 0.70))
        n_val = max(1, round(n * 0.20)) if n >= 5 else 0
        n_train = min(n_train, n - n_val)
        splits["train"].extend(group[:n_train])
        splits["val"].extend(group[n_train:n_train + n_val])
        splits["test"].extend(group[n_train + n_val:])

    if output.exists():
        shutil.rmtree(output, ignore_errors=True)

    for split, members in splits.items():
        for index, (image, target) in enumerate(members):
            stem = f"{index:06d}_{image.stem}"
            destination = output / "images" / split / f"{stem}{image.suffix.lower()}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, destination)

            label_path = output / "labels" / split / f"{stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            
            rows = []
            if target is not None and target in name_to_id:
                # Full frame bounding box [class_id, x_center, y_center, width, height]
                rows.append(f"{name_to_id[target]} 0.500000 0.500000 1.000000 1.000000")
            label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    config = {
        "path": str(output.resolve().as_posix()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(classes)},
    }
    yaml_path = output / "dataset.yaml"
    yaml_path.write_text(yaml.dump(config, sort_keys=False), encoding="utf-8")

    report = {split: len(items) for split, items in splits.items()}
    report["classes"] = Counter(target or "healthy_negative" for _, target in records)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    
    print(f"[+] Dataset materialized successfully at: '{yaml_path}'")
    print(f"    - Splits: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
    print(f"    - Classes: {len(classes)} pest & disease categories")
    return yaml_path


def check_gpu_environment(requested_device: str | int | None) -> str | int:
    """Validate GPU environment and display details using Ultralytics checks."""
    print("=" * 60)
    print("      YOLO26 ULTRALYTICS GPU & ENVIRONMENT CHECK")
    print("=" * 60)
    try:
        check_yolo()
    except Exception as e:
        print(f"Ultralytics environment check notice: {e}")

    chosen_device = 0 if requested_device is None else requested_device
    print(f"Target Device     : {chosen_device}")
    print("=" * 60)
    return chosen_device


def prepare_and_validate_dataset(data_path: Path, auto_generate: bool = True) -> Path:
    """Validate dataset configuration file and resolve relative paths to absolute paths."""
    resolved_data_path = data_path.resolve()

    if not resolved_data_path.is_file():
        alt_path = Path("yolo_dataset/dataset.yaml").resolve()
        if alt_path.is_file():
            resolved_data_path = alt_path
        elif auto_generate and Path("dataset").is_dir():
            return generate_pseudo_yolo_dataset(Path("dataset").resolve(), Path("yolo_dataset").resolve())
        else:
            sys.exit(f"Error: Dataset configuration file not found at '{data_path}' or '{alt_path}'.")

    with resolved_data_path.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    # Determine absolute path of dataset root
    raw_path = data_cfg.get("path", ".")
    raw_path_obj = Path(raw_path)

    if raw_path_obj.is_absolute():
        dataset_root = raw_path_obj
    else:
        dataset_root = (resolved_data_path.parent / raw_path_obj).resolve()

    val_rel = data_cfg.get("val", "images/val")
    val_dir = dataset_root / val_rel

    if not val_dir.exists():
        if auto_generate and Path("dataset").is_dir():
            print(f"\n[!] Notice: Detection dataset missing at '{val_dir}'. Auto-generating 'yolo_dataset' from 'dataset/'...")
            return generate_pseudo_yolo_dataset(Path("dataset").resolve(), Path("yolo_dataset").resolve())
        else:
            print(f"\n[!] ERROR: Dataset images directory not found at: '{val_dir}'")
            print("    The object detection dataset ('yolo_dataset') has not been materialized yet.")
            print("\n    To build the object detection dataset from your raw images and LabelMe annotations, run:")
            print("      python prepare_yolo11_dataset.py --source dataset --annotations annotations --output yolo_dataset\n")
            sys.exit(1)

    # Ensure path in data_cfg is absolute so Ultralytics does not look into standard cache dirs
    data_cfg["path"] = str(dataset_root.as_posix())
    resolved_yaml_path = resolved_data_path.parent / f"_resolved_{resolved_data_path.name}"
    resolved_yaml_path.write_text(yaml.dump(data_cfg, sort_keys=False), encoding="utf-8")
    return resolved_yaml_path


def calculate_f1(precision: float, recall: float) -> float:
    """Calculate F1-Score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def print_and_save_metrics(metrics, save_dir: Path) -> dict:
    """Extract, display, and format evaluation metrics (mAP, Precision, Recall, F1, Accuracy)."""
    box_metrics = metrics.box
    map50 = float(box_metrics.map50)
    map50_95 = float(box_metrics.map)
    mp = float(box_metrics.mp)
    mr = float(box_metrics.mr)
    f1 = calculate_f1(mp, mr)
    fitness = float(metrics.fitness)

    summary_metrics = {
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "Precision": round(mp, 4),
        "Recall": round(mr, 4),
        "F1_Score": round(f1, 4),
        "Fitness": round(fitness, 4),
    }

    print("\n" + "=" * 60)
    print("        YOLO26 OBJECT DETECTION EVALUATION SCORES")
    print("=" * 60)
    print(f"  mAP@50      (Overall Accuracy @ IoU 0.50) : {summary_metrics['mAP50']:.4f} ({summary_metrics['mAP50'] * 100:.2f}%)")
    print(f"  mAP@50-95   (Overall Accuracy @ IoU 0.50:0.95) : {summary_metrics['mAP50_95']:.4f} ({summary_metrics['mAP50_95'] * 100:.2f}%)")
    print(f"  Precision   (Positive Predictive Value)  : {summary_metrics['Precision']:.4f} ({summary_metrics['Precision'] * 100:.2f}%)")
    print(f"  Recall      (Sensitivity / True Pos Rate): {summary_metrics['Recall']:.4f} ({summary_metrics['Recall'] * 100:.2f}%)")
    print(f"  F1-Score    (Harmonic Mean P & R)        : {summary_metrics['F1_Score']:.4f} ({summary_metrics['F1_Score'] * 100:.2f}%)")
    print(f"  Fitness     (Combined Metric Score)      : {summary_metrics['Fitness']:.4f}")
    print("=" * 60)

    per_class_scores = []
    if hasattr(box_metrics, "maps") and len(box_metrics.maps) > 0 and hasattr(metrics, "names"):
        print("\nPer-Class Breakdown:")
        print(f"{'Class ID':<10} {'Class Name':<35} {'mAP50':<10} {'mAP50-95':<10}")
        print("-" * 68)
        
        p_class = getattr(box_metrics, "p", [None] * len(metrics.names))
        r_class = getattr(box_metrics, "r", [None] * len(metrics.names))
        
        for idx, map_val in enumerate(box_metrics.maps):
            cls_name = metrics.names.get(idx, f"class_{idx}")
            cls_map50 = float(box_metrics.class_result(idx)[2]) if hasattr(box_metrics, "class_result") else 0.0
            cls_map50_95 = float(map_val)
            
            p_val = float(p_class[idx]) if idx < len(p_class) and p_class[idx] is not None else None
            r_val = float(r_class[idx]) if idx < len(r_class) and r_class[idx] is not None else None
            
            class_entry = {
                "class_id": idx,
                "class_name": cls_name,
                "mAP50": round(cls_map50, 4),
                "mAP50_95": round(cls_map50_95, 4),
            }
            if p_val is not None:
                class_entry["precision"] = round(p_val, 4)
            if r_val is not None:
                class_entry["recall"] = round(r_val, 4)
                
            per_class_scores.append(class_entry)
            print(f"{idx:<10} {cls_name:<35} {cls_map50:<10.4f} {cls_map50_95:<10.4f}")
        print("-" * 68)

    full_results = {
        "model": "YOLO26",
        "summary": summary_metrics,
        "per_class": per_class_scores,
    }

    # Save summary json
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "yolo26_evaluation_results.json"
    save_path.write_text(json.dumps(full_results, indent=2), encoding="utf-8")
    print(f"\n[+] Detailed evaluation metrics saved to: [yolo26_evaluation_results.json](file:///{save_path.as_posix()})")

    return full_results


def main():
    parser = argparse.ArgumentParser(description="Train YOLO26 Object Detector for Rice & Maize Pest and Disease Detection.")
    parser.add_argument("--data", type=Path, default=Path("dataset.yaml"), help="Path to dataset.yaml (default: dataset.yaml)")
    parser.add_argument("--model", default="yolo26s.pt", help="Pretrained YOLO26 checkpoint (e.g., yolo26n.pt, yolo26s.pt, yolo26m.pt)")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs (default: 150)")
    parser.add_argument("--batch", type=int, default=2, help="Batch size (default: 2, safe for 6GB VRAM)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640, reduces memory usage)")
    parser.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision (default: disabled)")
    parser.add_argument("--device", default=None, help="GPU device ID (e.g. 0 or 0,1), cpu, or leave blank for auto GPU selection")
    parser.add_argument("--project", default="runs", help="Project output directory")
    parser.add_argument("--name", default="pest_disease_yolo26", help="Run experiment name")
    parser.add_argument("--patience", type=int, default=35, help="Early stopping patience epochs")
    parser.add_argument("--workers", type=int, default=0, help="Data loader worker threads (default: 0 on Windows)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--val-only", action="store_true", help="Skip training and run validation/evaluation on pre-trained model")
    parser.add_argument("--no-auto-dataset", action="store_true", help="Disable automatic dataset generation from raw images if yolo_dataset is missing")

    args = parser.parse_args()

    # Validate dataset configuration & resolve absolute path (auto-generates yolo_dataset from raw dataset/ if missing)
    data_path = prepare_and_validate_dataset(args.data, auto_generate=not args.no_auto_dataset)

    # GPU Environment check
    device = check_gpu_environment(args.device)

    # Initialize YOLO26 model
    print(f"\n[+] Loading YOLO26 model weights: {args.model}")
    model = YOLO(args.model)

    if not args.val_only:
        print(f"[+] Starting YOLO26 Object Detection Training on GPU ({device})...")
        print(f"    - Dataset Config : {data_path}")
        print(f"    - Image Size     : {args.imgsz}")
        print(f"    - Epochs         : {args.epochs}")
        print(f"    - Batch Size     : {args.batch}")

        # Execute GPU Training
        model.train(
            data=str(data_path.as_posix()),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            project=args.project,
            name=args.name,
            pretrained=True,
            patience=args.patience,
            cache=False,
            optimizer="auto",
            cos_lr=True,
            mosaic=0.35,
            close_mosaic=20,
            mixup=0.05,
            copy_paste=0.15,
            degrees=8,
            translate=0.08,
            scale=0.35,
            fliplr=0.5,
            multi_scale=True,
            workers=args.workers,
            seed=args.seed,
            plots=True,
        )

        save_dir = Path(model.trainer.save_dir)
        best_weights = save_dir / "weights" / "best.pt"
        print(f"\n[+] Training complete! Best weights saved to: [best.pt](file:///{best_weights.as_posix()})")

        # Load best trained model weights for evaluation
        if best_weights.exists():
            model = YOLO(str(best_weights))

    # Perform Validation and compute accuracy & scores
    print("\n[+] Evaluating YOLO26 Model on Validation Dataset...")
    val_results = model.val(data=str(data_path.as_posix()), imgsz=args.imgsz, device=device, split="val")

    save_dir = Path(getattr(model.trainer, "save_dir", Path(args.project) / args.name))
    print_and_save_metrics(val_results, save_dir)


if __name__ == "__main__":
    main()
