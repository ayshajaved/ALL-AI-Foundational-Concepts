# One-Stage Detectors

> **Speed First** - YOLO, SSD, and RetinaNet

---

## ⚡ YOLO (You Only Look Once) - 2016

**Idea:** Treat detection as a **regression problem**.
Divide image into an $S \times S$ grid.
If an object's center falls in a grid cell, that cell is responsible for detecting it.

**Output Tensor:** $S \times S \times (B \cdot 5 + C)$
- $B$: Number of boxes per cell.
- $5$: $x, y, w, h, \text{confidence}$.
- $C$: Class probabilities.

**Evolution:**
- **YOLOv1:** Fast (45 FPS) but poor localization.
- **YOLOv3:** Added Multi-Scale detection (FPN) and Darknet-53.
- **YOLOv5/v8:** PyTorch native, Anchor-free (v8), Mosaic Augmentation.

---

## 🧱 SSD (Single Shot MultiBox Detector)

**Idea:** Detect objects at **multiple scales** using different feature map layers.
- Early layers (high res) detect small objects.
- Deep layers (low res) detect large objects.

---

## 👁️ RetinaNet & Focal Loss

**Problem:** One-stage detectors suffer from **Class Imbalance**.
Thousands of background anchors (Easy Negatives) overwhelm the few object anchors (Hard Positives).

**Solution:** **Focal Loss**.
Down-weights easy examples so the model focuses on hard ones.

$$ FL(p_t) = -(1 - p_t)^\gamma \log(p_t) $$

- If $p_t \approx 1$ (Easy), term $\to 0$.
- If $p_t \approx 0$ (Hard), term $\approx 1$.

---

## 💻 Ultralytics YOLOv8

The industry standard for practical detection.

```python
from ultralytics import YOLO

# 1. Load Model
model = YOLO("yolov8n.pt") # Nano model (fastest)

# 2. Train
# model.train(data="coco128.yaml", epochs=3)

# 3. Predict
results = model("bus.jpg")

# 4. Visualize
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    result.show()         # display to screen
    result.save(filename="result.jpg")
```

---

## 🎓 Interview Focus

1.  **YOLO vs Faster R-CNN?**
    - **YOLO:** Faster, simpler pipeline. Slightly lower accuracy on small objects (historically).
    - **Faster R-CNN:** More accurate, handles small objects better. Slower.

2.  **Why does YOLO struggle with small objects?**
    - In early versions, the grid was coarse. If two small objects fell into the same grid cell, it could only detect one.
    - Modern YOLO (v8) uses FPN (Feature Pyramid Networks) to solve this.

3.  **Explain Focal Loss intuition.**
    - "Stop learning from the background sky! Focus on that tiny bird you missed."

---

**YOLO: Real-time vision for the masses!**
