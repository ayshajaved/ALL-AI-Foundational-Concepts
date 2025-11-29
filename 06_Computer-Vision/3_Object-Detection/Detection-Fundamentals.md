# Object Detection Fundamentals

> **Where is it?** - IoU, NMS, mAP, and Anchors

---

## 🎯 The Task

**Classification:** "There is a cat in this image."
**Detection:** "There is a cat at `[x, y, w, h]`."

Output: List of Bounding Boxes `(Class, Confidence, X, Y, W, H)`.

---

## 📏 Intersection over Union (IoU)

Metric to measure overlap between two boxes (Predicted vs Ground Truth).

$$ \text{IoU} = \frac{\text{Area of Intersection}}{\text{Area of Union}} $$

- **IoU > 0.5:** Decent match.
- **IoU > 0.7:** Good match.
- **IoU > 0.9:** Perfect match.

```python
def calculate_iou(box1, box2):
    # box = [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    return intersection / union
```

---

## 🧹 Non-Maximum Suppression (NMS)

Detectors often predict multiple boxes for the same object.
**Goal:** Keep only the best one.

**Algorithm:**
1.  Sort boxes by Confidence Score.
2.  Pick the highest confidence box $B$. Add to final list.
3.  Remove all other boxes that have high IoU (e.g., > 0.5) with $B$.
4.  Repeat until empty.

---

## ⚓ Anchor Boxes (Priors)

Instead of predicting width/height from scratch (which is hard), we predict **offsets** from pre-defined boxes.
- **Anchors:** Set of boxes with different scales (small, medium, large) and aspect ratios (1:1, 1:2, 2:1) placed at every pixel.
- **Prediction:** $\Delta x, \Delta y, \Delta w, \Delta h$ relative to the anchor.

---

## 📈 Mean Average Precision (mAP)

The gold standard metric.
1.  Calculate **Precision-Recall Curve** for each class at a specific IoU threshold (e.g., 0.5).
2.  **AP (Average Precision):** Area under the PR curve.
3.  **mAP:** Mean of AP across all classes.
4.  **mAP@0.5:0.95:** Average mAP over IoU thresholds from 0.5 to 0.95 (COCO metric).

---

## 🎓 Interview Focus

1.  **Why use Anchors?**
    - It stabilizes training. It's easier for a network to say "Make this box 10% wider" than "Predict a box of width 153px".

2.  **What happens if NMS threshold is too low?**
    - You might delete valid boxes of *other* objects that are close to the main object (e.g., a person holding a cat).

3.  **Precision vs Recall in Detection?**
    - **High Precision:** Few false positives (only detects sure things).
    - **High Recall:** Few false negatives (detects everything, even garbage).

---

**Fundamentals: The language of bounding boxes!**
