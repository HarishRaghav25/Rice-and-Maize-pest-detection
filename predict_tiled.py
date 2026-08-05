"""High-resolution tiled YOLO11 prediction for small pests in large photos."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0, x2-x1) * max(0, y2-y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - overlap
    return overlap / max(1, union)


def main():
    parser = argparse.ArgumentParser(description="Tile large images before YOLO11 inference.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tile", type=int, default=1280)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--output", type=Path, default=Path("runs/tiled_predictions"))
    args = parser.parse_args()
    if not 0 <= args.overlap < 1: raise SystemExit("--overlap must be in [0, 1)")
    model, image = YOLO(args.weights), Image.open(args.source).convert("RGB")
    step, boxes = max(1, int(args.tile * (1 - args.overlap))), []
    for top in range(0, image.height, step):
        for left in range(0, image.width, step):
            crop = image.crop((left, top, min(left + args.tile, image.width), min(top + args.tile, image.height)))
            result = model(crop, imgsz=args.tile, conf=args.conf, verbose=False)[0]
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist(); score = float(box.conf[0]); cls = int(box.cls[0])
                boxes.append(([x1+left, y1+top, x2+left, y2+top], score, cls))
    boxes.sort(key=lambda item: item[1], reverse=True)
    kept = []
    for candidate in boxes:
        if not any(candidate[2] == chosen[2] and iou(candidate[0], chosen[0]) >= 0.5 for chosen in kept): kept.append(candidate)
    draw = ImageDraw.Draw(image)
    names = model.names
    for (x1, y1, x2, y2), score, cls in kept:
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw.text((x1, max(0, y1 - 16)), f"{names[cls]} {score:.2f}", fill="red")
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / f"{args.source.stem}_detections.jpg"
    image.save(target)
    print(f"Saved {len(kept)} detections to {target}")


if __name__ == "__main__":
    main()
