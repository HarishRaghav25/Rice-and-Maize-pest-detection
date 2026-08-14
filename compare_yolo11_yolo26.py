"""Compare YOLO11s and YOLO26s inference latency, model size, and detection outputs."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from PIL import Image, ImageDraw
import torch
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Compare YOLO11s and YOLO26s models.")
    parser.add_argument(
        "--model11",
        default="yolo11s.pt",
        help="Path to YOLO11 weights (default: yolo11s.pt)"
    )
    parser.add_argument(
        "--model26",
        default="runs/detect/runs/pest_disease_yolo26s-2/weights/best.pt",
        help="Path to YOLO26 weights (default: runs/detect/runs/pest_disease_yolo26s-2/weights/best.pt)"
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("yolo_dataset/images/val/000000_1b8efd49-9702-47f7-b7e8-3f3857a0a45d.jpg"),
        help="Path to test image"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warm-up iterations"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=50,
        help="Number of timed inference iterations"
    )
    parser.add_argument(
        "--output",
        default="runs/comparison_output.jpg",
        help="Path to save visual comparison"
    )
    args = parser.parse_args()

    # Verify files exist
    m11_path = Path(args.model11)
    m26_path = Path(args.model26)
    
    # Fallbacks if custom weights don't exist
    if not m11_path.is_file():
        m11_path = Path("yolo11s.pt")
        print(f"[!] Warning: Custom YOLO11 weights not found, using default '{m11_path}'")
        
    if not m26_path.is_file():
        m26_path = Path("yolo26s.pt")
        print(f"[!] Warning: Custom YOLO26 weights not found, using default '{m26_path}'")
        
    if not args.image.is_file():
        # Search for any image in val split
        val_imgs = list(Path("yolo_dataset/images/val").glob("*.jpg"))
        if val_imgs:
            args.image = val_imgs[0]
            print(f"[!] Target image not found, using fallback: '{args.image}'")
        else:
            raise SystemExit(f"Error: Target image '{args.image}' not found, and no fallbacks exist.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print(f"DEVICE: {device.upper()}")
    if device == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    # 1. Load Models
    print(f"\n[+] Loading YOLO11: {m11_path}...")
    model11 = YOLO(str(m11_path))
    
    print(f"[+] Loading YOLO26: {m26_path}...")
    model26 = YOLO(str(m26_path))

    # 2. Extract Model Stats
    def get_model_params(model):
        try:
            # Get number of parameters (in millions)
            params = sum(p.numel() for p in model.model.parameters()) / 1e6
            return f"{params:.2f}M"
        except Exception:
            return "N/A"

    params_11 = get_model_params(model11)
    params_26 = get_model_params(model26)

    # 3. Latency Benchmarking Function
    def benchmark_model(model, image_path, warmup_runs, test_runs):
        # Warmup
        for _ in range(warmup_runs):
            _ = model(image_path, verbose=False)
            
        if device == "cuda":
            torch.cuda.synchronize()
            
        start_time = time.time()
        for _ in range(test_runs):
            _ = model(image_path, verbose=False)
            
        if device == "cuda":
            torch.cuda.synchronize()
            
        end_time = time.time()
        avg_latency = ((end_time - start_time) / test_runs) * 1000  # in ms
        return avg_latency

    print(f"\n[+] Benchmarking YOLO11 latency ({args.runs} runs)...")
    latency_11 = benchmark_model(model11, args.image, args.warmup, args.runs)
    
    print(f"[+] Benchmarking YOLO26 latency ({args.runs} runs)...")
    latency_26 = benchmark_model(model26, args.image, args.warmup, args.runs)

    # 4. Get Bounding Box Counts and Scores
    res11 = model11(args.image, verbose=False)[0]
    res26 = model26(args.image, verbose=False)[0]
    
    boxes_11 = len(res11.boxes)
    boxes_26 = len(res26.boxes)

    # 5. Visual Rendering (Side-by-Side)
    orig_img = Image.open(args.image).convert("RGB")
    w, h = orig_img.size
    
    # Scale width down if giant, to save image size
    max_dim = 1024
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img11 = orig_img.resize((new_w, new_h))
        img26 = orig_img.resize((new_w, new_h))
    else:
        new_w, new_h = w, h
        img11 = orig_img.copy()
        img26 = orig_img.copy()
        scale = 1.0

    draw11 = ImageDraw.Draw(img11)
    draw26 = ImageDraw.Draw(img26)
    
    # Draw YOLO11 Bounding Boxes
    for box in res11.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        score = float(box.conf[0])
        cls = int(box.cls[0])
        name = model11.names[cls]
        
        # Scale coordinates
        x1, y1, x2, y2 = x1*scale, y1*scale, x2*scale, y2*scale
        draw11.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw11.text((x1, max(0, y1-15)), f"{name} {score:.2f}", fill="red")
        
    # Draw YOLO26 Bounding Boxes
    for box in res26.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        score = float(box.conf[0])
        cls = int(box.cls[0])
        name = model26.names[cls]
        
        # Scale coordinates
        x1, y1, x2, y2 = x1*scale, y1*scale, x2*scale, y2*scale
        draw26.rectangle((x1, y1, x2, y2), outline="blue", width=3)
        draw26.text((x1, max(0, y1-15)), f"{name} {score:.2f}", fill="blue")

    # Combine side-by-side
    combined_img = Image.new("RGB", (new_w * 2, new_h + 100), "white")
    combined_img.paste(img11, (0, 100))
    combined_img.paste(img26, (new_w, 100))
    
    # Draw Labels at the top of the combined image
    top_draw = ImageDraw.Draw(combined_img)
    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
    except Exception:
        font = None
        
    top_draw.text((20, 20), f"YOLO11s | Parameters: {params_11} | Latency: {latency_11:.2f}ms | Detections: {boxes_11}", fill="black")
    top_draw.text((new_w + 20, 20), f"YOLO26s | Parameters: {params_26} | Latency: {latency_26:.2f}ms | Detections: {boxes_26}", fill="black")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_img.save(output_path)
    
    print("\n" + "=" * 60)
    print("            COMPARATIVE EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Metric              YOLO11s                  YOLO26s")
    print("-" * 60)
    print(f"  Weights File        {m11_path.name:<24} {m26_path.name:<24}")
    print(f"  Parameters          {params_11:<24} {params_26:<24}")
    print(f"  Inference Latency   {latency_11:.2f} ms / image         {latency_26:.2f} ms / image")
    print(f"  Detections Count    {boxes_11:<24} {boxes_26:<24}")
    print("=" * 60)
    print(f"[+] Saved comparison visual to: [{args.output}](file:///{output_path.resolve().as_posix()})")

if __name__ == "__main__":
    main()
