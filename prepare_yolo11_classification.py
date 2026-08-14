"""Split the existing crop-folder dataset for Ultralytics YOLO11 classification."""
from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def slug(value: str) -> str:
    value = re.sub(r"^\d+[_ -]*", "", value.lower())
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def class_name(image: Path, source: Path) -> str:
    """Map the supplied crop/type/folder structure to an unambiguous class name."""
    parts = image.relative_to(source).parts
    
    if len(parts) == 3 and parts[1].lower() == "healthy":
        return f"{slug(parts[0])}_healthy"
    
    if len(parts) != 4:
        raise ValueError(f"Unsupported source layout: {image.relative_to(source)}")
    type_map = {"disease": "disease", "insect-pests": "pest"}
    category = type_map.get(parts[1].lower())
    if category is None:
        raise ValueError(f"Expected Disease or Insect-pests folder: {image.relative_to(source)}")
    return f"{slug(parts[0])}_{category}_{slug(parts[2])}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create stratified folders for YOLO11 image classification.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("classification_dataset"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    
    if output.exists():
        raise SystemExit(f"Output already exists: {output}. Choose a new path or remove it deliberately.")
    
    images = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    
    if not images:
        raise SystemExit(f"No images found in {source}")
    
    groups: dict[str, list[Path]] = {}
    
    for image in images:
        groups.setdefault(class_name(image, source), []).append(image)

    rng = random.Random(args.seed)
    counts = Counter()

    try:
        for name, members in sorted(groups.items()):
            rng.shuffle(members)
            total = len(members)
            train_end = max(1, round(total * 0.70))
            val_end = min(total, train_end + max(1, round(total * 0.20)))

            for split, subset in (("train", members[:train_end]), ("val", members[train_end:val_end]), ("test", members[val_end:])):

                for index, image in enumerate(subset):
                    destination = output / split / name / f"{index:04d}_{image.name}"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(image, destination)
                    counts[split] += 1
        
        metadata = {
            "task": "classification",
            "classes": sorted(groups),
            "class_counts": {name: len(members) for name, members in sorted(groups.items())},
            "splits": dict(counts),
        }
        
        (output / "classes.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    
    print(f"Created {output}")
    print(f"Classes ({len(groups)}): {', '.join(sorted(groups))}")
    print(f"Images: {dict(counts)}")


if __name__ == "__main__":
    main()
