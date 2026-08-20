"""Fix invalid YOLO label coordinates and augment the small dataset.

This script:
1. Clamps all bounding box coordinates to valid [0, 1] range
2. Removes boxes that are too small (< 0.5% of image area)
3. Creates offline augmented copies to boost training images by 3-4x
4. Regenerates clean dataset with fixed labels

Run BEFORE training:
    python fix_and_augment_dataset.py
"""
from __future__ import annotations

import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

INPUT_DIR = Path("rice_annotated_dataset")
OUTPUT_DIR = Path("rice_dataset_fixed")

MIN_BOX_AREA = 0.0025  # Minimum box area as fraction of image (0.25%)
NUM_CLASSES = 8

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


def clamp_label(line: str) -> str | None:
    """Clamp YOLO box coordinates to [0, 1] and validate.
    
    Returns fixed line or None if box is too small / invalid.
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    
    cid = int(parts[0])
    cx, cy, w, h = map(float, parts[1:5])
    
    # Compute absolute corners
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    
    # Clamp to [0, 1]
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    
    # Recompute center + size
    new_w = x2 - x1
    new_h = y2 - y1
    new_cx = (x1 + x2) / 2
    new_cy = (y1 + y2) / 2
    
    # Skip tiny boxes
    if new_w * new_h < MIN_BOX_AREA:
        return None
    if new_w < 0.01 or new_h < 0.01:
        return None
    
    return f"{cid} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}"


def fix_label_file(src: Path, dst: Path) -> tuple[int, int, int]:
    """Fix all labels in a file. Returns (original, fixed, dropped) counts."""
    content = src.read_text(encoding="utf-8").strip()
    if not content:
        dst.write_text("", encoding="utf-8")
        return (0, 0, 0)
    
    lines = content.split("\n")
    fixed_lines = []
    dropped = 0
    
    for line in lines:
        if not line.strip():
            continue
        fixed = clamp_label(line)
        if fixed is not None:
            fixed_lines.append(fixed)
        else:
            dropped += 1
    
    dst.write_text("\n".join(fixed_lines) + ("\n" if fixed_lines else ""), encoding="utf-8")
    return (len(lines), len(fixed_lines), dropped)


def augment_image_and_labels(
    img_path: Path, 
    lbl_path: Path, 
    out_img_dir: Path, 
    out_lbl_dir: Path, 
    aug_idx: int
) -> bool:
    """Create one augmented copy of an image with transformed labels."""
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    
    h, w = img.shape[:2]
    
    # Read labels
    content = lbl_path.read_text(encoding="utf-8").strip()
    if not content:
        return False
    
    rng = random.Random(aug_idx * 1000 + hash(img_path.name) % 10000)
    
    # Choose augmentation type
    aug_type = aug_idx % 4
    
    if aug_type == 0:
        # Horizontal flip
        img_aug = cv2.flip(img, 1)
        new_lines = []
        for line in content.split("\n"):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid = parts[0]
            cx, cy, bw, bh = map(float, parts[1:5])
            new_cx = 1.0 - cx
            new_lines.append(f"{cid} {new_cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    
    elif aug_type == 1:
        # Brightness + Contrast jitter
        alpha = rng.uniform(0.7, 1.3)  # Contrast
        beta = rng.randint(-30, 30)     # Brightness
        img_aug = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        new_lines = content.split("\n")
    
    elif aug_type == 2:
        # HSV color jitter (simulate different lighting)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-10, 10)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.7, 1.3), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * rng.uniform(0.7, 1.3), 0, 255)
        img_aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        new_lines = content.split("\n")
    
    else:
        # Vertical flip
        img_aug = cv2.flip(img, 0)
        new_lines = []
        for line in content.split("\n"):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid = parts[0]
            cx, cy, bw, bh = map(float, parts[1:5])
            new_cy = 1.0 - cy
            new_lines.append(f"{cid} {cx:.6f} {new_cy:.6f} {bw:.6f} {bh:.6f}")
    
    # Save augmented image and label
    stem = f"aug{aug_idx}_{img_path.stem}"
    suffix = img_path.suffix.lower()
    
    cv2.imwrite(str(out_img_dir / f"{stem}{suffix}"), img_aug)
    lbl_content = "\n".join(l.strip() for l in new_lines if l.strip()) + "\n"
    (out_lbl_dir / f"{stem}.txt").write_text(lbl_content, encoding="utf-8")
    
    return True


def main():
    if OUTPUT_DIR.exists():
        print(f"[!] Removing existing output: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    
    print("=" * 60)
    print("  PHASE 1: Fix Invalid Bounding Box Coordinates")
    print("=" * 60)
    
    total_original = 0
    total_fixed = 0
    total_dropped = 0
    
    for split in ["train", "val", "test"]:
        img_src = INPUT_DIR / "images" / split
        lbl_src = INPUT_DIR / "labels" / split
        img_dst = OUTPUT_DIR / "images" / split
        lbl_dst = OUTPUT_DIR / "labels" / split
        
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)
        
        split_orig = 0
        split_fixed = 0
        split_dropped = 0
        
        for img_file in sorted(img_src.glob("*")):
            if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            
            lbl_file = lbl_src / f"{img_file.stem}.txt"
            
            # Copy image
            shutil.copy2(img_file, img_dst / img_file.name)
            
            # Fix labels
            if lbl_file.exists():
                orig, fixed, dropped = fix_label_file(lbl_file, lbl_dst / lbl_file.name)
                split_orig += orig
                split_fixed += fixed
                split_dropped += dropped
            else:
                (lbl_dst / f"{img_file.stem}.txt").write_text("", encoding="utf-8")
        
        total_original += split_orig
        total_fixed += split_fixed
        total_dropped += split_dropped
        print(f"  {split}: {split_orig} boxes -> {split_fixed} fixed, {split_dropped} dropped")
    
    print(f"\n  TOTAL: {total_original} -> {total_fixed} fixed, {total_dropped} dropped")
    
    print(f"\n{'=' * 60}")
    print("  PHASE 2: Offline Augmentation (Training Set Only)")
    print("=" * 60)
    
    # Count per-class boxes in training set
    train_lbl_dir = OUTPUT_DIR / "labels" / "train"
    train_img_dir = OUTPUT_DIR / "images" / "train"
    
    class_files: dict[int, list[tuple[Path, Path]]] = {i: [] for i in range(NUM_CLASSES)}
    
    for lbl_file in sorted(train_lbl_dir.glob("*.txt")):
        content = lbl_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        
        img_candidates = list(train_img_dir.glob(f"{lbl_file.stem}.*"))
        if not img_candidates:
            continue
        img_file = img_candidates[0]
        
        for line in content.split("\n"):
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(parts[0])
                if cid in class_files:
                    class_files[cid].append((img_file, lbl_file))
                    break  # Count file once per its primary class
    
    # Target: at least 80 training images per class
    TARGET_PER_CLASS = 80
    total_augmented = 0
    
    for cid in range(NUM_CLASSES):
        files = class_files[cid]
        current = len(files)
        needed = max(0, TARGET_PER_CLASS - current)
        
        if needed == 0:
            print(f"  class {cid} ({CLASS_NAMES[cid]}): {current} images, no augmentation needed")
            continue
        
        aug_count = 0
        for aug_idx in range(needed):
            src_img, src_lbl = files[aug_idx % len(files)]
            success = augment_image_and_labels(
                src_img, src_lbl, train_img_dir, train_lbl_dir, aug_idx
            )
            if success:
                aug_count += 1
        
        total_augmented += aug_count
        print(f"  class {cid} ({CLASS_NAMES[cid]}): {current} -> {current + aug_count} images (+{aug_count} augmented)")
    
    print(f"\n  Total augmented images added: {total_augmented}")
    
    # Recount final stats
    print(f"\n{'=' * 60}")
    print("  PHASE 3: Generate dataset.yaml and Final Stats")
    print("=" * 60)
    
    import yaml
    
    config = {
        "path": str(OUTPUT_DIR.resolve().as_posix()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": CLASS_NAMES,
    }
    yaml_path = OUTPUT_DIR / "dataset.yaml"
    yaml_path.write_text(yaml.dump(config, sort_keys=False), encoding="utf-8")
    
    for split in ["train", "val", "test"]:
        img_count = len(list((OUTPUT_DIR / "images" / split).glob("*")))
        box_count = 0
        counter = Counter()
        for lbl in (OUTPUT_DIR / "labels" / split).glob("*.txt"):
            content = lbl.read_text().strip()
            if not content:
                counter["NEGATIVE"] += 1
                continue
            for line in content.split("\n"):
                parts = line.strip().split()
                if len(parts) >= 5:
                    box_count += 1
                    counter[int(parts[0])] += 1
        
        print(f"\n  {split.upper()}: {img_count} images, {box_count} boxes")
        for cid in sorted([k for k in counter if isinstance(k, int)]):
            print(f"    class {cid} ({CLASS_NAMES[cid]}): {counter[cid]}")
        neg = counter.get("NEGATIVE", 0)
        if neg:
            print(f"    NEGATIVE/HEALTHY: {neg}")
    
    print(f"\n  dataset.yaml: {yaml_path}")
    print(f"\n{'=' * 60}")
    print("  DONE! Use this for training:")
    print(f"  python train_yolo11.py --data {yaml_path} --imgsz 480 --batch 4")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
