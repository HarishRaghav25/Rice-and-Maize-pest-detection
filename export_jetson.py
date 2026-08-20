"""Export trained YOLO models to ONNX/TensorRT for Jetson Nano 2GB deployment.

This script exports the best-trained checkpoint to formats optimised for
the NVIDIA Jetson Nano 2GB developer kit (Maxwell GPU, ARM CPU).

Usage
-----
    # Export to ONNX (recommended -- works on all Jetson variants)
    python export_jetson.py --weights runs/detect/pest_disease_rice_yolo26s/weights/best.pt --format onnx

    # Export to TensorRT (must run on Jetson or compatible GPU)
    python export_jetson.py --weights runs/detect/pest_disease_rice_yolo26s/weights/best.pt --format engine

    # Export at smaller image size for real-time inference
    python export_jetson.py --weights runs/detect/pest_disease_rice_yolo26s/weights/best.pt --imgsz 256 --format onnx
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ultralytics import YOLO


# Recommended image sizes for Jetson Nano 2GB
JETSON_IMGSZ = {
    "realtime": 256,    # ~15-20 FPS on Jetson Nano 2GB
    "balanced": 320,    # ~8-12 FPS on Jetson Nano 2GB
    "accuracy": 416,    # ~3-5 FPS on Jetson Nano 2GB
}


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLO model for Jetson Nano deployment."
    )
    parser.add_argument(
        "--weights", type=Path, required=True,
        help="Path to trained best.pt checkpoint",
    )
    parser.add_argument(
        "--format", choices=["onnx", "engine", "torchscript", "tflite"],
        default="onnx",
        help="Export format (onnx recommended for Jetson Nano)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=320,
        help="Input image size for export (256=realtime, 320=balanced, 416=accuracy)",
    )
    parser.add_argument(
        "--half", action="store_true", default=True,
        help="Export with FP16 half precision (default: enabled for Jetson)",
    )
    parser.add_argument(
        "--int8", action="store_true",
        help="Export with INT8 quantisation (requires calibration data)",
    )
    parser.add_argument(
        "--dynamic", action="store_true",
        help="Enable dynamic input shapes in ONNX",
    )
    parser.add_argument(
        "--simplify", action="store_true", default=True,
        help="Simplify ONNX graph (default: enabled)",
    )
    parser.add_argument(
        "--opset", type=int, default=12,
        help="ONNX opset version (12 works with JetPack 4.6)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device for export (cpu recommended for ONNX, 0 for TensorRT)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("jetson_models"),
        help="Directory to copy exported model into",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run a quick benchmark after export",
    )
    args = parser.parse_args()

    if not args.weights.is_file():
        raise SystemExit(f"Weights not found: {args.weights}")

    print("=" * 60)
    print("  YOLO MODEL EXPORT FOR JETSON NANO")
    print("=" * 60)
    print(f"  Weights    : {args.weights}")
    print(f"  Format     : {args.format.upper()}")
    print(f"  Image size : {args.imgsz}")
    print(f"  FP16 half  : {args.half}")
    print(f"  INT8       : {args.int8}")
    print(f"  Device     : {args.device}")
    print("=" * 60)

    # Load model
    model = YOLO(str(args.weights))

    # Build export kwargs
    export_kwargs = {
        "format": args.format,
        "imgsz": args.imgsz,
        "device": args.device,
    }

    if args.format == "onnx":
        export_kwargs["simplify"] = args.simplify
        export_kwargs["dynamic"] = args.dynamic
        export_kwargs["opset"] = args.opset
        if args.half:
            export_kwargs["half"] = True

    elif args.format == "engine":
        export_kwargs["half"] = args.half
        if args.int8:
            export_kwargs["int8"] = True
        export_kwargs["device"] = 0  # TensorRT needs GPU

    elif args.format == "torchscript":
        pass  # Default settings are fine

    # Export
    print(f"\n[+] Exporting model to {args.format.upper()}...")
    exported_path = model.export(**export_kwargs)
    print(f"[+] Export complete: {exported_path}")

    # Copy to output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exported_file = Path(exported_path)

    if exported_file.is_file():
        dest = args.output_dir / exported_file.name
        import shutil
        shutil.copy2(exported_file, dest)
        file_size_mb = dest.stat().st_size / (1024 * 1024)

        print(f"\n[+] Model copied to: {dest}")
        print(f"[+] Model size: {file_size_mb:.2f} MB")

        # Save export metadata
        metadata = {
            "source_weights": str(args.weights),
            "export_format": args.format,
            "imgsz": args.imgsz,
            "half": args.half,
            "int8": args.int8,
            "model_size_mb": round(file_size_mb, 2),
            "exported_file": str(dest),
            "class_names": model.names,
            "recommended_jetson_settings": {
                "power_mode": "MAXN (10W) for best performance",
                "input_size": args.imgsz,
                "expected_fps_jetson_nano_4gb": JETSON_IMGSZ.get(
                    {320: "realtime", 416: "balanced", 640: "accuracy"}.get(args.imgsz, "custom"),
                    "varies"
                ),
            },
        }
        meta_path = args.output_dir / "export_metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"[+] Metadata saved to: {meta_path}")

    # Benchmark if requested
    if args.benchmark and exported_file.is_file():
        print(f"\n[+] Running benchmark on exported model...")
        try:
            benchmark_model = YOLO(str(exported_file))
            import time
            import numpy as np

            # Warmup
            dummy = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)
            for _ in range(5):
                benchmark_model(dummy, verbose=False)

            # Timed runs
            n_runs = 20
            start = time.time()
            for _ in range(n_runs):
                benchmark_model(dummy, verbose=False)
            elapsed = time.time() - start

            avg_ms = (elapsed / n_runs) * 1000
            fps = 1000 / avg_ms

            print(f"  Average latency : {avg_ms:.1f} ms/image")
            print(f"  Throughput      : {fps:.1f} FPS")
            print(f"  (Note: Jetson Nano will be ~3-5x slower than desktop GPU)")
        except Exception as e:
            print(f"  Benchmark skipped: {e}")

    # Print Jetson Nano deployment instructions
    print(f"\n{'=' * 60}")
    print(f"  JETSON NANO DEPLOYMENT INSTRUCTIONS")
    print(f"{'=' * 60}")
    print(f"""
  1. Copy these files to your Jetson Nano 2GB:
     - {args.output_dir / exported_file.name}
     - jetson_inference.py
     - jetson_setup.sh

  2. On the Jetson Nano 2GB, run the setup script:
     chmod +x jetson_setup.sh && ./jetson_setup.sh

  3. Run inference:
     python3 jetson_inference.py --model {exported_file.name} --source image.jpg
     python3 jetson_inference.py --model {exported_file.name} --source 0  # USB camera

  4. For best performance on Jetson Nano 2GB:
     sudo nvpmodel -m 0
     sudo jetson_clocks

  5. If you run out of memory, add swap:
     sudo fallocate -l 4G /var/swapfile
     sudo chmod 600 /var/swapfile
     sudo mkswap /var/swapfile
     sudo swapon /var/swapfile
""")


if __name__ == "__main__":
    main()
