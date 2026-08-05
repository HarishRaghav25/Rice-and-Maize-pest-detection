"""Train YOLO11 classification on the existing rice/maize folder dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO11 pest/disease image classifier.")
    parser.add_argument("--data", type=Path, default=Path("classification_dataset"), help="Folder with train/, val/, and test/ class folders")
    parser.add_argument("--model", default="yolo11s-cls.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default=None, help="Use 0 for first NVIDIA GPU, cpu, or omit for auto")
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="pest_disease_yolo11_classify")
    args = parser.parse_args()
    data = args.data.resolve()
    missing = [data / split for split in ("train", "val") if not (data / split).is_dir()]
    if missing:
        raise SystemExit("Classification data is missing. Run prepare_yolo11_classification.py first. Missing: " + ", ".join(map(str, missing)))
    model = YOLO(args.model)
    model.train(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
        project=args.project, name=args.name, pretrained=True, patience=25, optimizer="auto",
        cos_lr=True, fliplr=0.5, translate=0.08, scale=0.25, erasing=0.15,
        workers=4, seed=42, plots=True,
    )
    print(f"Training complete. Best checkpoint: {model.trainer.save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
