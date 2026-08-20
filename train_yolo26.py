"""Train a YOLO26s detector for rice pest and disease detection.

Trained on manually annotated Label Studio bounding boxes (8 rice classes).
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


def box_iou(box1, box2):
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    xi1 = max(x11, x21)
    yi1 = max(y11, y21)
    xi2 = min(x12, x22)
    yi2 = min(y12, y22)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)
    union_area = box1_area + box2_area - inter_area
    return inter_area / max(1e-6, union_area)


def compute_ap_pure_python(tp, fp, num_gts):
    if num_gts == 0:
        return 0.0
    tp_cum = []
    fp_cum = []
    t_sum = 0
    f_sum = 0
    for t, f in zip(tp, fp):
        t_sum += t
        f_sum += f
        tp_cum.append(t_sum)
        fp_cum.append(f_sum)
        
    recalls = [t / num_gts for t in tp_cum]
    precisions = [t / (t + f) if (t + f) > 0 else 0.0 for t, f in zip(tp_cum, fp_cum)]
    
    mrec = [0.0] + recalls + [1.0]
    mpre = [1.0] + precisions + [0.0]
    
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
        
    ap = 0.0
    for i in range(len(mrec) - 1):
        if mrec[i+1] != mrec[i]:
            ap += (mrec[i+1] - mrec[i]) * mpre[i+1]
    return ap


def run_tiled_evaluation(model, data_cfg_path: Path, tile_size: int, overlap: float, device: str | int | None, save_dir: Path, conf: float = 0.001):
    import yaml
    from PIL import Image
    
    with data_cfg_path.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
        
    dataset_root = Path(data_cfg["path"])
    val_rel_img = data_cfg.get("val", "images/val")
    val_img_dir = (dataset_root / val_rel_img).resolve()
    
    val_label_dir = Path(str(val_img_dir).replace("images", "labels"))
    
    class_names = data_cfg.get("names", {})
    class_names = {int(k): v for k, v in class_names.items()}
    
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(p for p in val_img_dir.glob("*") if p.suffix.lower() in img_extensions)
    
    if not images:
        print(f"[!] No validation images found in: {val_img_dir}")
        return
        
    print(f"\n[+] Running Tiled Validation on {len(images)} images (Tile Size: {tile_size}, Overlap: {overlap})...")
    
    all_gts = {cid: 0 for cid in class_names}
    pred_by_class = {cid: [] for cid in class_names}
    
    step = max(1, int(tile_size * (1 - overlap)))
    
    for idx, img_path in enumerate(images):
        label_path = val_label_dir / f"{img_path.stem}.txt"
        gts = []
        if label_path.is_file():
            content = label_path.read_text(encoding="utf-8").strip()
            if content:
                for line in content.split("\n"):
                    if not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) == 5:
                        cid = int(parts[0])
                        gts.append((cid, float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
                        
        with Image.open(img_path) as img:
            w, h = img.size
            
        abs_gts = []
        for cid, x, y, bw, bh in gts:
            x1 = (x - bw/2) * w
            y1 = (y - bh/2) * h
            x2 = (x + bw/2) * w
            y2 = (y + bh/2) * h
            abs_gts.append((cid, x1, y1, x2, y2))
            all_gts[cid] = all_gts.get(cid, 0) + 1
            
        pil_img = Image.open(img_path).convert("RGB")
        img_w, img_h = pil_img.size
        
        raw_boxes = []
        for top in range(0, img_h, step):
            for left in range(0, img_w, step):
                crop_w = min(tile_size, img_w - left)
                crop_h = min(tile_size, img_h - top)
                if crop_w <= 0 or crop_h <= 0:
                    continue
                crop = pil_img.crop((left, top, left + crop_w, top + crop_h))
                results = model(crop, imgsz=tile_size, conf=conf, device=device, verbose=False)[0]
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    score = float(box.conf[0])
                    cls = int(box.cls[0])
                    raw_boxes.append(([x1 + left, y1 + top, x2 + left, y2 + top], score, cls))
                    
        raw_boxes.sort(key=lambda x: x[1], reverse=True)
        kept_preds = []
        for cand in raw_boxes:
            cand_box, cand_score, cand_cls = cand
            if not any(cand_cls == chosen[2] and box_iou(cand_box, chosen[0]) >= 0.5 for chosen in kept_preds):
                kept_preds.append(cand)
                
        preds_by_cid = {cid: [] for cid in class_names}
        for box, score, cls in kept_preds:
            if cls in preds_by_cid:
                preds_by_cid[cls].append((score, box))
                
        gts_by_cid = {cid: [] for cid in class_names}
        for cid, x1, y1, x2, y2 in abs_gts:
            gts_by_cid[cid].append([x1, y1, x2, y2, False])
            
        for cid in class_names:
            p_list = preds_by_cid[cid]
            g_list = gts_by_cid[cid]
            
            p_list.sort(key=lambda x: x[0], reverse=True)
            
            for score, p_box in p_list:
                best_iou = 0.0
                best_gt_idx = -1
                for g_idx, g_box in enumerate(g_list):
                    iou_val = box_iou(p_box, g_box[:4])
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_gt_idx = g_idx
                        
                if best_iou >= 0.5 and best_gt_idx != -1 and not g_list[best_gt_idx][4]:
                    g_list[best_gt_idx][4] = True
                    pred_by_class[cid].append((score, 1, 0))
                else:
                    pred_by_class[cid].append((score, 0, 1))

    per_class_results = []
    overall_tp = 0
    overall_fp = 0
    overall_gt = sum(all_gts.values())
    
    for cid in class_names:
        name = class_names[cid]
        c_preds = pred_by_class[cid]
        c_preds.sort(key=lambda x: x[0], reverse=True)
        
        tp = [x[1] for x in c_preds]
        fp = [x[2] for x in c_preds]
        num_gts = all_gts[cid]
        
        ap50 = compute_ap_pure_python(tp, fp, num_gts)
        
        total_tp = sum(tp)
        total_fp = sum(fp)
        overall_tp += total_tp
        overall_fp += total_fp
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / num_gts if num_gts > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_results.append({
            "class_id": cid,
            "class_name": name,
            "mAP50": round(ap50, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": num_gts
        })
        
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
    overall_recall = overall_tp / overall_gt if overall_gt > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    valid_aps = [r["mAP50"] for r in per_class_results if r["support"] > 0]
    overall_map50 = sum(valid_aps) / len(valid_aps) if valid_aps else 0.0
    
    summary_metrics = {
        "mAP50": round(overall_map50, 4),
        "Precision": round(overall_precision, 4),
        "Recall": round(overall_recall, 4),
        "F1_Score": round(overall_f1, 4),
    }
    
    print("\n" + "=" * 60)
    print("      YOLO26 TILED OBJECT DETECTION EVALUATION SCORES")
    print("=" * 60)
    print(f"  mAP@50      (Overall Accuracy @ IoU 0.50) : {summary_metrics['mAP50']:.4f} ({summary_metrics['mAP50'] * 100:.2f}%)")
    print(f"  Precision   (Positive Predictive Value)  : {summary_metrics['Precision']:.4f} ({summary_metrics['Precision'] * 100:.2f}%)")
    print(f"  Recall      (Sensitivity / True Pos Rate): {summary_metrics['Recall']:.4f} ({summary_metrics['Recall'] * 100:.2f}%)")
    print(f"  F1-Score    (Harmonic Mean P & R)        : {summary_metrics['F1_Score']:.4f} ({summary_metrics['F1_Score'] * 100:.2f}%)")
    print("=" * 60)
    
    print("\nTiled Per-Class Breakdown:")
    print(f"{'Class ID':<10} {'Class Name':<35} {'mAP50':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 90)
    for r in per_class_results:
        print(f"{r['class_id']:<10} {r['class_name']:<35} {r['mAP50']:<10.4f} {r['precision']:<10.4f} {r['recall']:<10.4f} {r['f1']:<10.4f}")
    print("-" * 90)
    
    full_results = {
        "model": "YOLO26_Tiled",
        "summary": summary_metrics,
        "per_class": per_class_results
    }
    
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "yolo26_tiled_evaluation_results.json"
    save_path.write_text(json.dumps(full_results, indent=2), encoding="utf-8")
    print(f"\n[+] Tiled validation results saved to: [yolo26_tiled_evaluation_results.json](file:///{save_path.as_posix()})")
    return full_results


def main():
    parser = argparse.ArgumentParser(description="Train YOLO26s Object Detector for Rice Pest and Disease Detection.")
    parser.add_argument("--data", type=Path, default=Path("rice_annotated_dataset/dataset.yaml"), help="Path to dataset.yaml (default: rice_annotated_dataset/dataset.yaml)")
    parser.add_argument("--model", default="yolo26s.pt", help="Pretrained YOLO26 checkpoint (e.g., yolo26n.pt, yolo26s.pt, yolo26m.pt)")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs (default: 300 for full convergence)")
    parser.add_argument("--batch", type=int, default=2, help="Batch size (minimum 2 for BatchNorm; use larger with more VRAM)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable Automatic Mixed Precision (default: enabled)")
    parser.add_argument("--device", default=None, help="GPU device ID (e.g. 0 or 0,1), cpu, or leave blank for auto GPU selection")
    parser.add_argument("--project", default="runs", help="Project output directory")
    parser.add_argument("--name", default="pest_disease_rice_yolo26s", help="Run experiment name")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience epochs")
    parser.add_argument("--workers", type=int, default=0, help="Data loader worker threads (default: 0 on Windows)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--val-only", action="store_true", help="Skip training and run validation/evaluation on pre-trained model")
    parser.add_argument("--no-auto-dataset", action="store_true", help="Disable automatic dataset generation from raw images if yolo_dataset is missing")
    parser.add_argument("--tile-val", action="store_true", help="Run tiled/sliced validation instead of standard validation")
    parser.add_argument("--tile-size", type=int, default=1280, help="Tile size for tiled validation/inference")
    parser.add_argument("--tile-overlap", type=float, default=0.25, help="Overlap ratio between tiles")
    parser.add_argument("--tile-conf", type=float, default=0.001, help="Confidence threshold for tiled validation")

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

        # Execute GPU Training with accuracy-optimised hyperparameters
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
            amp=args.amp,

            # --- Convergence ---
            optimizer="AdamW",         # Better convergence on small datasets
            lr0=0.001,                 # Initial learning rate
            lrf=0.01,                  # Final LR factor
            cos_lr=True,               # Cosine annealing schedule
            warmup_epochs=5,           # Warmup for stable start
            warmup_momentum=0.5,       # Lower momentum during warmup
            weight_decay=0.0005,       # L2 regularisation

            # --- Regularisation ---
            label_smoothing=0.1,       # Prevents overconfident predictions
            dropout=0.1,               # Dropout in classification head

            # --- Augmentation (tuned for pest/disease data) ---
            cache=False,
            mosaic=0.8,                # Strong mosaic
            close_mosaic=25,           # Disable mosaic last 25 epochs
            mixup=0.1,                 # Moderate mixup
            copy_paste=0.1,            # Copy-paste augmentation
            degrees=15,                # Rotation range
            translate=0.1,             # Translation range
            scale=0.5,                 # Scale variation
            shear=2.0,                 # Slight shear
            fliplr=0.5,                # Horizontal flip
            flipud=0.1,                # Occasional vertical flip
            hsv_h=0.02,                # Hue shift (subtle — preserve colours)
            hsv_s=0.7,                 # Saturation variation
            hsv_v=0.4,                 # Brightness variation
            multi_scale=False,         # Disabled: prevents OOM with small batch on 6GB GPU

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

    save_dir = Path(getattr(model.trainer, "save_dir", Path(args.project) / args.name))

    if args.tile_val:
        run_tiled_evaluation(
            model=model,
            data_cfg_path=data_path,
            tile_size=args.tile_size,
            overlap=args.tile_overlap,
            device=device,
            save_dir=save_dir,
            conf=args.tile_conf
        )
    else:
        # Perform Standard Validation and compute accuracy & scores
        print("\n[+] Evaluating YOLO26 Model on Validation Dataset...")
        val_results = model.val(data=str(data_path.as_posix()), imgsz=args.imgsz, device=device, split="val")
        print_and_save_metrics(val_results, save_dir)


if __name__ == "__main__":
    main()
