"""Train a YOLO11s detector for rice pest and disease detection.

Trained on manually annotated Label Studio bounding boxes (8 rice classes).
Optimised for high validation accuracy (target: 90%+ mAP@50) with:
- AdamW optimiser with cosine annealing LR schedule
- Label smoothing and dropout regularisation
- Agricultural-tuned augmentation (subtle HSV, strong mosaic)
- 300 epochs with patience 50 for full convergence
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO11s for rice pest and disease detection."
    )

    parser.add_argument(
        "--data", type=Path, default=Path("rice_annotated_dataset/dataset.yaml"),
        help="YOLO dataset.yaml (default: rice_annotated_dataset/dataset.yaml)",
    )
    parser.add_argument(
        "--model", default="yolo11s.pt",
        help="Pretrained YOLO11 checkpoint (yolo11s.pt or yolo11m.pt)",
    )
    parser.add_argument("--epochs", type=int, default=300,
                        help="Training epochs (default: 300 for full convergence)")
    parser.add_argument("--imgsz", type=int, default=480,
                        help="Input image size (480 for small dataset, increase to 640 with more data)")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size (4 improves BatchNorm stability on small datasets)")
    parser.add_argument("--device", default=None,
                        help="GPU id (0), cpu, or omit for auto")
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="pest_disease_rice_yolo11s")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")

    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"Dataset configuration not found: {args.data}")

    model = YOLO(args.model)

    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,

        # --- Transfer learning ---
        pretrained=True,

        # --- Convergence ---
        patience=50,              # Early stopping: wait 50 epochs for improvement
        optimizer="AdamW",        # AdamW converges better on small datasets
        lr0=0.001,                # Initial learning rate
        lrf=0.01,                 # Final LR = lr0 * lrf
        cos_lr=True,              # Cosine annealing schedule
        warmup_epochs=5,          # 5-epoch warmup for stable start
        warmup_momentum=0.5,      # Lower momentum during warmup
        weight_decay=0.0005,      # L2 regularisation

        # --- Regularisation ---
        label_smoothing=0.1,      # Prevents overconfident predictions
        dropout=0.1,              # Dropout in classification head

        # --- Augmentation (tuned for SMALL agricultural pest data) ---
        mosaic=0.5,               # Moderate mosaic (reduced: small dataset)
        close_mosaic=30,          # Disable mosaic for last 30 epochs
        mixup=0.0,                # Disabled: hurts with very few images
        copy_paste=0.0,           # Disabled: needs larger dataset
        degrees=10,               # Reduced rotation range
        translate=0.1,            # Translation range
        scale=0.3,                # Reduced scale variation
        shear=2.0,                # Slight shear
        fliplr=0.5,               # Horizontal flip
        flipud=0.0,               # Disabled: pest orientation matters
        hsv_h=0.015,              # Subtle hue shift
        hsv_s=0.5,                # Moderate saturation variation
        hsv_v=0.3,                # Moderate brightness variation

        # --- Performance ---
        cache=False,              # Disable cache (saves RAM)
        workers=0,                # 0 on Windows to prevent cv2 OOM
        amp=True,                 # Mixed precision for speed + larger batch
        multi_scale=False,        # Disabled: prevents OOM with small batch on 6GB GPU
        seed=42,
        plots=True,
    )

    save_dir = Path(model.trainer.save_dir)
    best_wt = save_dir / "weights" / "best.pt"
    print(f"\n[+] Training complete! Best checkpoint: {best_wt}")

    # Run validation on best weights
    print("\n[+] Running final validation on best weights...")
    best_model = YOLO(str(best_wt))
    val_results = best_model.val(
        data=str(args.data.resolve()),
        imgsz=args.imgsz,
        device=args.device,
        split="val",
        workers=0,   # 0 = no subprocesses; prevents cv2 OOM on Windows
    )

    # Print summary
    box = val_results.box
    map50 = float(box.map50)
    map50_95 = float(box.map)
    mp = float(box.mp)
    mr = float(box.mr)
    f1 = 2 * mp * mr / (mp + mr) if (mp + mr) > 0 else 0.0

    print(f"\n{'=' * 60}")
    print(f"  YOLO11s VALIDATION RESULTS")
    print(f"{'=' * 60}")
    print(f"  mAP@50         : {map50:.4f} ({map50 * 100:.2f}%)")
    print(f"  mAP@50-95      : {map50_95:.4f} ({map50_95 * 100:.2f}%)")
    print(f"  Precision      : {mp:.4f} ({mp * 100:.2f}%)")
    print(f"  Recall         : {mr:.4f} ({mr * 100:.2f}%)")
    print(f"  F1-Score       : {f1:.4f} ({f1 * 100:.2f}%)")
    status = "PASSED" if map50 >= 0.90 else "BELOW TARGET"
    print(f"  Target (>90%)  : {status}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
