# Rice Pest & Disease Detection (YOLO11s & YOLO26s)

ICAR project for detecting rice pests and diseases in field images using
**YOLOv11s** and **YOLOv26s** object detection models, with deployment
support for **NVIDIA Jetson Nano 2GB**.

## Dataset Overview

| Property | Value |
|:---|:---|
| **Total images** | 403 (manually annotated via Label Studio) |
| **Bounding boxes** | 761 (tight, manually drawn) |
| **Classes** | 8 rice pest & disease categories |
| **Healthy negatives** | 50 background images |
| **Splits** | 70% train / 20% val / 10% test |

## Quick Start — Full Pipeline

```powershell
# Step 1: Build the YOLO dataset from Label Studio annotations
python build_annotated_dataset.py

# Step 2: Train YOLOv11s (300 epochs, ~6-8 hours on RTX 4050)
python train_yolo11.py

# Step 3: Train YOLOv26s (300 epochs, ~6-8 hours on RTX 4050)
python train_yolo26.py

# Step 4: Export best model for Jetson Nano 2GB
python export_jetson.py --weights runs/pest_disease_rice_yolo26s/weights/best.pt --format onnx
```

## 1. Build Dataset from Annotations

The raw annotations are in `annotated dataset/rice/` with per-folder Label Studio
exports. The build script remaps all local class IDs to a unified 8-class numbering:

```powershell
python build_annotated_dataset.py --source "annotated dataset" --output rice_annotated_dataset
```

This generates:
- `rice_annotated_dataset/images/{train,val,test}/` — images
- `rice_annotated_dataset/labels/{train,val,test}/` — YOLO-format `.txt` labels
- `rice_annotated_dataset/dataset.yaml` — Ultralytics dataset config
- `rice_annotated_dataset/report.json` — class distribution stats

## 2. Train YOLO Detectors

### YOLOv11s

```powershell
python train_yolo11.py --epochs 300 --device 0
```

### YOLOv26s

```powershell
python train_yolo26.py --epochs 300 --device 0
```

**Key training optimisations:**
- AdamW optimiser with cosine annealing LR schedule
- 5-epoch warmup for stable convergence
- Label smoothing (0.1) to prevent overconfident predictions
- Dropout (0.1) in classification head
- Strong mosaic, mixup, and HSV colour augmentations
- Early stopping with patience=50

## 3. Tiled Inference for Small Pests

For high-resolution field images where pests are small:

```powershell
python predict_tiled.py --weights runs/pest_disease_rice_yolo11s/weights/best.pt --source field_photo.jpg
```

## 4. Jetson Nano 2GB Deployment

### Export the Model

```powershell
# ONNX (recommended — universal compatibility)
python export_jetson.py --weights runs/pest_disease_rice_yolo26s/weights/best.pt --format onnx

# TensorRT (run on Jetson for optimal speed)
python export_jetson.py --weights runs/pest_disease_rice_yolo26s/weights/best.pt --format engine --imgsz 256
```

### On the Jetson Nano 2GB

```bash
# 1. Setup (includes 4GB swap for 2GB kit)
chmod +x jetson_setup.sh && ./jetson_setup.sh

# 2. Run inference on an image
python3 jetson_inference.py --model jetson_models/best.onnx --source field_photo.jpg

# 3. Live camera inference
python3 jetson_inference.py --model jetson_models/best.onnx --source 0        # USB camera
python3 jetson_inference.py --model jetson_models/best.onnx --source csi      # CSI camera

# 4. For best performance:
sudo nvpmodel -m 0 && sudo jetson_clocks
```

### Expected Performance on Jetson Nano 2GB

| Image Size | Expected FPS | Use Case |
|:---|:---|:---|
| 256 | ~15-20 FPS | Real-time field scanning |
| 320 | ~8-12 FPS | Balanced speed/accuracy |
| 416 | ~3-5 FPS | Maximum detection accuracy |

## Detection Classes

| ID | Class Name | Category |
|:---|:---|:---|
| 0 | rice_disease_bacterial_leaf_blight | Disease |
| 1 | rice_disease_brown_spot | Disease |
| 2 | rice_disease_false_smut | Disease |
| 3 | rice_disease_leaf_sheath_blight | Disease |
| 4 | rice_pest_leaf_folder | Pest |
| 5 | rice_pest_rice_skipper | Pest |
| 6 | rice_pest_white_stem_borer | Pest |
| 7 | rice_pest_yellow_stem_borer | Pest |

## File Structure

```
├── build_annotated_dataset.py   # Build YOLO dataset from Label Studio annotations
├── train_yolo11.py              # Train YOLOv11s (optimised for rice)
├── train_yolo26.py              # Train YOLOv26s (optimised for rice)
├── validate_models.py           # Validate and compare both models
├── predict_tiled.py             # Tiled inference for large images
├── export_jetson.py             # Export to ONNX/TensorRT for Jetson Nano 2GB
├── jetson_inference.py          # Standalone inference on Jetson Nano 2GB
├── jetson_setup.sh              # Jetson Nano 2GB environment setup
├── prepare_yolo11_dataset.py    # (Legacy) Dataset prep with LabelMe annotations
├── annotated dataset/           # Raw Label Studio annotations (rice only)
├── rice_annotated_dataset/      # Built YOLO detection dataset (8 classes)
└── runs/                        # Training outputs and checkpoints
```

## Hardware Requirements

| Component | Training | Deployment |
|:---|:---|:---|
| **GPU** | NVIDIA RTX 4050 (6GB VRAM) | NVIDIA Jetson Nano 2GB |
| **RAM** | 16GB+ | 2GB + 4GB swap |
| **Python** | 3.12 | 3.6+ |
| **Framework** | Ultralytics 8.4+ / PyTorch 2.11+ | ONNX Runtime |
