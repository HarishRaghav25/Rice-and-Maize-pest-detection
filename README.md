# Rice and maize pest + disease detection (YOLO11)

This project trains an **object detector**, not an image classifier. The current
`dataset/` folders provide image-level class names, but do not include bounding
boxes. Box annotations are mandatory for meaningful pest detection—especially
for aphids and other small insects. Do not use a full-image box as a substitute.

## 1. Annotate the images

Use Label Studio, CVAT, Roboflow, or LabelMe to draw a tight box around every
visible pest, lesion, or disease symptom. Export LabelMe JSON files and place
them in a mirrored `annotations/` tree, for example:

```text
annotations/Rice/Insect-pests/05_Leaf_folder/4c6ee278-....json
dataset/Rice/Insect-pests/05_Leaf_folder/4c6ee278-....Jpg
```

Each LabelMe shape must be a rectangle and its `label` must be one of the class
names written by the preparation command. Healthy images are retained as
negative examples and require no JSON file.

## 2. Create a detection dataset

```powershell
python -m pip install -r requirements.txt
python prepare_yolo11_dataset.py --source dataset --annotations annotations --output yolo_dataset
```

The command creates `yolo_dataset/dataset.yaml`, deterministic train/validation/test
splits, and a report. It stops if an affected image has no annotation, preventing
accidental training with unlabeled pests.

## 3. Train YOLO Detector (YOLO11 or YOLO26)

### Train YOLO26 (Recommended)

```powershell
python train_yolo26.py --data dataset.yaml --model yolo26n.pt --epochs 150 --device 0
```

`yolo26n.pt` (or `yolo26s.pt`) is used with GPU training enabled (`--device 0`). The script automatically evaluates validation accuracy and metrics, computing:
- **mAP@50** (Overall Detection Accuracy @ IoU 0.50)
- **mAP@50-95** (Overall Detection Accuracy @ IoU 0.50:0.95)
- **Precision**, **Recall**, and **F1-Score**
- Per-class metric breakdown saved to `yolo26_evaluation_results.json`.

### Train YOLO11

```powershell
python train_yolo11.py --data yolo_dataset/dataset.yaml --model yolo11s.pt --epochs 150 --device 0
```

`yolo11s.pt` is a good baseline model. The program uses 1280-pixel input and a
small-object augmentation profile. If GPU memory permits, use `yolo11m.pt`.

## 4. Detect with tiling

```powershell
python predict_tiled.py --weights runs/pest_disease_yolo11/weights/best.pt --source path\\to\\field_photo.jpg
```

Tiling prevents a tiny insect in a wide field image from being reduced to only
a few pixels. Outputs are written to `runs/tiled_predictions/`.

## Train now with the supplied folder-only data (classification)

The original `dataset/` directory has image-level labels, so it can train a
classifier immediately. This predicts the condition of an entire photo; it
does **not** locate a pest or disease with a bounding box.

```powershell
python prepare_yolo11_classification.py --source dataset --output classification_dataset
python train_yolo11_classify.py --data classification_dataset --model yolo11s-cls.pt --epochs 100 --device 0
```

## Labels for this dataset

Run the preparation command first and use the printed canonical labels exactly.
They combine crop, target type, and folder name (for example
`rice_pest_leaf_folder` and `maize_disease_maydis_leaf_blight`). Healthy images
are background-only images, not an object class.
