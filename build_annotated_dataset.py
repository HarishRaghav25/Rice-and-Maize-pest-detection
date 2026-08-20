"""Build a unified YOLO detection dataset from manually annotated Label Studio exports.

Each Label Studio project exported its own local class IDs (starting from 0).
This script reads per-folder ``classes.txt`` / ``notes.json`` metadata, remaps
every local class ID to a single global numbering, copies images and remapped
label files into a clean ``rice_annotated_dataset/`` directory, and generates
``dataset.yaml`` + ``report.json``.

Usage
-----
    python build_annotated_dataset.py
    python build_annotated_dataset.py --source "annotated dataset" --output rice_annotated_dataset
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ─── Unified global class map ───────────────────────────────────────────
# Each entry maps (source_folder_name, local_class_id) -> global_class_id.
# Sub-labels from the same pest are merged into a single class.

GLOBAL_CLASSES: dict[int, str] = {
    0: "rice_disease_bacterial_leaf_blight",
    1: "rice_disease_brown_spot",
    2: "rice_disease_false_smut",
    3: "rice_disease_leaf_sheath_blight",
    4: "rice_pest_leaf_folder",
    5: "rice_pest_rice_skipper",
    6: "rice_pest_white_stem_borer",
    7: "rice_pest_yellow_stem_borer",
}

# Mapping: folder_key -> { local_class_id: global_class_id }
# folder_key is the leaf folder name inside disease/ or insect_pests/
REMAP: dict[str, dict[int, int]] = {
    # ── Diseases (each project had a single class at local ID 0) ──
    "bacterial_leaf_blight": {0: 0},
    "brown_spot": {0: 1},
    "false_smut": {0: 2},
    "leaf_sheath_blight": {0: 3},
    # ── Pests ──
    "leaf_folder": {
        0: 4,  # leaf_folder_pest
        1: 4,  # rolled_leaf -> merged
    },
    "rice_skipper": {
        0: 5,  # rice_skipper
        1: 5,  # rice_skipper_egg -> merged
        2: 5,  # rice_skipper_larvae -> merged
    },
    "stem_borer_family": {
        0: 6,  # defected_plants -> white_stem_borer
        1: 6,  # larvae -> white_stem_borer
        2: 6,  # white_stem_borer
        3: 7,  # yellow_stem_borer
    },
}


def discover_annotated_folders(source: Path) -> list[tuple[Path, str]]:
    """Return (folder_path, folder_key) for every annotated subfolder."""
    results: list[tuple[Path, str]] = []

    for category_dir in sorted(source.iterdir()):
        if not category_dir.is_dir():
            continue
        name = category_dir.name.lower()

        if name == "healthy":
            # Healthy images — no subfolder structure, images directly here
            results.append((category_dir, "healthy"))
            continue

        if name in ("disease", "insect_pests"):
            for sub_dir in sorted(category_dir.iterdir()):
                if sub_dir.is_dir() and sub_dir.name in REMAP:
                    results.append((sub_dir, sub_dir.name))

    return results


def remap_label_file(
    label_path: Path, folder_key: str, *, strict: bool = True
) -> list[str]:
    """Read a YOLO label file and remap local class IDs to global IDs.

    Returns list of remapped YOLO-format lines.
    """
    mapping = REMAP.get(folder_key)
    if mapping is None:
        raise ValueError(f"No remap entry for folder key '{folder_key}'")

    lines: list[str] = []
    raw = label_path.read_text(encoding="utf-8").strip()
    if not raw:
        return lines

    for line_no, line in enumerate(raw.split("\n"), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            if strict:
                raise ValueError(
                    f"{label_path}:{line_no}: expected ≥5 fields, got {len(parts)}"
                )
            continue

        local_id = int(parts[0])
        if local_id not in mapping:
            if strict:
                raise ValueError(
                    f"{label_path}:{line_no}: local class ID {local_id} not in "
                    f"remap for '{folder_key}' (valid: {list(mapping)})"
                )
            continue

        global_id = mapping[local_id]
        lines.append(f"{global_id} {' '.join(parts[1:])}")

    return lines


def build_dataset(
    source: Path,
    output: Path,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
) -> Path:
    """Build the unified YOLO dataset from annotated source."""
    if output.exists():
        print(f"[!] Removing existing output directory: {output}")
        shutil.rmtree(output)

    # Discover annotated folders
    rice_root = source / "rice"
    if not rice_root.is_dir():
        sys.exit(f"Error: expected 'rice/' directory inside '{source}'")

    folders = discover_annotated_folders(rice_root)
    if not folders:
        sys.exit(f"Error: no annotated folders found in '{rice_root}'")

    print(f"[+] Found {len(folders)} annotated folders in '{rice_root}'")

    # Collect all records: (image_path, label_lines_or_none, global_class_or_none)
    records: list[tuple[Path, list[str] | None, int | None]] = []

    for folder_path, folder_key in folders:
        if folder_key == "healthy":
            # Healthy images: no labels, act as negative background
            images = sorted(
                p
                for p in folder_path.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            )
            for img in images:
                records.append((img, None, None))
            print(f"    {folder_key}: {len(images)} images (healthy/negative)")
            continue

        # Annotated folder with images/ and labels/ subdirectories
        img_dir = folder_path / "images"
        lbl_dir = folder_path / "labels"

        if not img_dir.is_dir() or not lbl_dir.is_dir():
            print(f"    [!] Skipping {folder_key}: missing images/ or labels/")
            continue

        images = sorted(
            p
            for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )

        for img in images:
            lbl_path = lbl_dir / f"{img.stem}.txt"
            if not lbl_path.is_file():
                print(f"    [!] Warning: no label for {img.name} in {folder_key}")
                records.append((img, [], None))
                continue

            remapped_lines = remap_label_file(lbl_path, folder_key, strict=True)

            # Determine the primary global class (for stratification)
            primary_class = None
            if remapped_lines:
                class_ids = [int(l.split()[0]) for l in remapped_lines]
                primary_class = Counter(class_ids).most_common(1)[0][0]

            records.append((img, remapped_lines, primary_class))

        print(
            f"    {folder_key}: {len(images)} images"
            f" -> global class(es) {sorted(set(REMAP[folder_key].values()))}"
        )

    print(f"\n[+] Total records: {len(records)}")

    # ── Stratified split ──
    # Group by primary class for stratification
    groups: dict[int | None, list[tuple[Path, list[str] | None, int | None]]] = {}
    for record in records:
        groups.setdefault(record[2], []).append(record)

    rng = random.Random(seed)
    splits: dict[str, list[tuple[Path, list[str] | None, int | None]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for cls_id, members in sorted(groups.items(), key=lambda x: (x[0] is None, x[0])):
        rng.shuffle(members)
        n = len(members)
        n_train = max(1, round(n * train_ratio))
        n_val = max(1, round(n * val_ratio)) if n >= 5 else 0
        n_train = min(n_train, n - n_val)
        splits["train"].extend(members[:n_train])
        splits["val"].extend(members[n_train : n_train + n_val])
        splits["test"].extend(members[n_train + n_val :])

    # ── Write output ──
    for split_name, split_records in splits.items():
        for idx, (img_path, label_lines, _) in enumerate(split_records):
            stem = f"{idx:06d}_{img_path.stem}"
            suffix = img_path.suffix.lower()

            # Copy image
            dest_img = output / "images" / split_name / f"{stem}{suffix}"
            dest_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dest_img)

            # Write remapped label
            dest_lbl = output / "labels" / split_name / f"{stem}.txt"
            dest_lbl.parent.mkdir(parents=True, exist_ok=True)
            if label_lines:
                dest_lbl.write_text(
                    "\n".join(label_lines) + "\n", encoding="utf-8"
                )
            else:
                # Empty label file for healthy/negative images
                dest_lbl.write_text("", encoding="utf-8")

    # ── dataset.yaml ──
    config = {
        "path": str(output.resolve().as_posix()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": GLOBAL_CLASSES,
    }
    yaml_path = output / "dataset.yaml"
    yaml_path.write_text(yaml.dump(config, sort_keys=False), encoding="utf-8")

    # ── report.json ──
    split_counts = {s: len(r) for s, r in splits.items()}

    class_counter: Counter = Counter()
    for _, label_lines, _ in records:
        if label_lines:
            for line in label_lines:
                cid = int(line.split()[0])
                class_counter[GLOBAL_CLASSES.get(cid, f"class_{cid}")] += 1
        else:
            class_counter["healthy_negative"] += 1

    # Count total bounding boxes per split
    bbox_per_split: dict[str, int] = {}
    for split_name, split_records in splits.items():
        total_boxes = sum(
            len(ll) for _, ll, _ in split_records if ll
        )
        bbox_per_split[split_name] = total_boxes

    report = {
        "images": split_counts,
        "bounding_boxes": bbox_per_split,
        "class_distribution": dict(class_counter.most_common()),
        "num_classes": len(GLOBAL_CLASSES),
        "classes": GLOBAL_CLASSES,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"  ANNOTATED DATASET BUILT SUCCESSFULLY")
    print(f"{'=' * 60}")
    print(f"  Output    : {output.resolve()}")
    print(f"  Classes   : {len(GLOBAL_CLASSES)}")
    print(f"  Images    : Train={split_counts['train']}, Val={split_counts['val']}, Test={split_counts['test']}")
    print(f"  BBoxes    : Train={bbox_per_split['train']}, Val={bbox_per_split['val']}, Test={bbox_per_split['test']}")
    print(f"  YAML      : {yaml_path}")
    print(f"{'=' * 60}")

    print("\n  Per-class distribution:")
    for cls_name, count in class_counter.most_common():
        print(f"    {cls_name}: {count} annotations")

    return yaml_path


def main():
    parser = argparse.ArgumentParser(
        description="Build unified YOLO dataset from Label Studio annotated exports."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("annotated dataset"),
        help="Root of annotated dataset (contains rice/ folder)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rice_annotated_dataset"),
        help="Output YOLO dataset directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()

    if not source.is_dir():
        sys.exit(f"Error: source directory not found: {source}")

    build_dataset(source, output, seed=args.seed)


if __name__ == "__main__":
    main()
