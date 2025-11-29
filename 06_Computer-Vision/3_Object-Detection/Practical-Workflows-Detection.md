# Practical Workflows: Training YOLOv8

> **Custom Object Detection** - From Labeling to Inference

---

## 🛠️ The Project: Pothole Detection

**Goal:** Detect potholes in road images.
**Dataset:** Custom images labeled in YOLO format.

---

## 1. Data Preparation (YOLO Format)

Directory structure:
```
dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

**Label Format (`.txt`):**
`class_id x_center y_center width height` (Normalized 0-1)
`0 0.5 0.5 0.2 0.3`

**`data.yaml`:**
```yaml
path: /content/dataset
train: train/images
val: val/images

nc: 1
names: ['pothole']
```

---

## 2. Training (Ultralytics)

```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(
    data='dataset/data.yaml', 
    epochs=50, 
    imgsz=640, 
    batch=16,
    name='pothole_detector'
)
```

---

## 3. Validation & Inference

```python
# Validate
metrics = model.val()
print(f"mAP@50: {metrics.box.map50}")

# Inference on new image
results = model("road_trip.jpg")
results[0].save()  # saves to 'runs/detect/predict/road_trip.jpg'
```

---

## 4. Export for Deployment

Export to ONNX for fast CPU inference or TensorRT for GPU.

```python
success = model.export(format='onnx')
```

---

## 🧠 Tips for Better Results

1.  **Mosaic Augmentation:** YOLO uses this by default. Stitches 4 images together. Helps detect small objects and context.
2.  **Rectangular Training:** Instead of padding square images with black bars, valid batches are resized to rectangular shapes to minimize padding pixels. Speeds up training.
3.  **Hyperparameter Evolution:** YOLOv8 has a genetic algorithm to auto-tune LR, momentum, etc.

---

**You can now detect anything!**
