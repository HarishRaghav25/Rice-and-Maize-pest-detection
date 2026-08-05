"""Materialize an annotated rice/maize dataset in Ultralytics YOLO detection format."""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def slug(text: str) -> str:
    text = re.sub(r"^\d+[_ -]*", "", text.lower())
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def image_files(root: Path):
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def class_for(image: Path, source: Path) -> str | None:
    relative = image.relative_to(source)
    parts = relative.parts
    crop = slug(parts[0])
    if len(parts) == 3 and parts[1].lower() == "healthy":
        return None
    if len(parts) < 4 or parts[1].lower() not in {"disease", "insect-pests"}:
        raise ValueError(f"Unsupported source layout: {relative}")
    kind = "pest" if parts[1].lower() == "insect-pests" else "disease"
    return f"{crop}_{kind}_{slug(parts[2])}"


def labelme_boxes(json_path: Path, width: int, height: int, name_to_id: dict[str, int]):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    rows = []
    for shape in data.get("shapes", []):
        label = slug(str(shape.get("label", "")))
        if label not in name_to_id:
            valid = ", ".join(name_to_id)
            raise ValueError(f"{json_path}: unknown label '{label}'. Use one of: {valid}")
        if shape.get("shape_type", "rectangle") != "rectangle" or len(shape.get("points", [])) != 2:
            raise ValueError(f"{json_path}: only two-point rectangle shapes are supported")
        (x1, y1), (x2, y2) = shape["points"]
        x1, x2 = sorted((max(0, x1), min(width, x2)))
        y1, y2 = sorted((max(0, y1), min(height, y2)))
        bw, bh = x2 - x1, y2 - y1
        if bw < 2 or bh < 2:
            continue
        rows.append(f"{name_to_id[label]} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} {bw / width:.6f} {bh / height:.6f}")
    return rows


def split_by_class(records, seed: int):
    """Stratify positives; distribute negative examples with the same ratios."""
    groups: dict[str, list[tuple[Path, str | None]]] = {}
    for record in records:
        groups.setdefault(record[1] or "__negative__", []).append(record)
    rng = random.Random(seed)
    split = {"train": [], "val": [], "test": []}
    for group in groups.values():
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, round(n * 0.70))
        n_val = max(1, round(n * 0.20)) if n >= 5 else 0
        n_train = min(n_train, n - n_val)
        split["train"].extend(group[:n_train])
        split["val"].extend(group[n_train:n_train + n_val])
        split["test"].extend(group[n_train + n_val:])
    return split


def main():
    parser = argparse.ArgumentParser(description="Convert mirrored LabelMe annotations to a YOLO11 detection dataset.")
    parser.add_argument("--source", type=Path, required=True, help="Raw dataset directory")
    parser.add_argument("--annotations", type=Path, required=True, help="Mirrored LabelMe JSON annotation directory")
    parser.add_argument("--output", type=Path, required=True, help="New YOLO output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source, annotations, output = args.source.resolve(), args.annotations.resolve(), args.output.resolve()
    if output.exists():
        raise SystemExit(f"Output already exists: {output}. Choose a new path or remove it deliberately.")
    images = image_files(source)
    if not images:
        raise SystemExit(f"No images found under {source}")
    classes = sorted({c for p in images if (c := class_for(p, source))})
    name_to_id = {name: i for i, name in enumerate(classes)}
    print("Detection labels:")
    for name, idx in name_to_id.items():
        print(f"  {idx}: {name}")
    records = [(image, class_for(image, source)) for image in images]
    missing = []
    for image, target in records:
        if target is not None:
            json_path = annotations / image.relative_to(source).with_suffix(".json")
            if not json_path.exists():
                missing.append(json_path)
    if missing:
        sample = "\n  ".join(str(p) for p in missing[:10])
        raise SystemExit(f"Missing LabelMe boxes for {len(missing)} affected images. First missing:\n  {sample}")
    splits = split_by_class(records, args.seed)
    try:
        for split, members in splits.items():
            for index, (image, target) in enumerate(members):
                stem = f"{index:06d}_{image.stem}"
                destination = output / "images" / split / f"{stem}{image.suffix.lower()}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image, destination)
                label_path = output / "labels" / split / f"{stem}.txt"
                label_path.parent.mkdir(parents=True, exist_ok=True)
                rows = []
                if target is not None:
                    with Image.open(image) as im:
                        rows = labelme_boxes(annotations / image.relative_to(source).with_suffix(".json"), *im.size, name_to_id)
                    if not rows:
                        raise ValueError(f"No valid boxes in annotation for affected image: {image}")
                label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    config = {"path": str(output), "train": "images/train", "val": "images/val", "test": "images/test", "names": {i: n for i, n in enumerate(classes)}}
    (output / "dataset.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    report = {split: len(items) for split, items in splits.items()}
    report["classes"] = Counter(target or "healthy_negative" for _, target in records)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Created {output} with splits: {dict((k, len(v)) for k, v in splits.items())}")


if __name__ == "__main__":
    main()
