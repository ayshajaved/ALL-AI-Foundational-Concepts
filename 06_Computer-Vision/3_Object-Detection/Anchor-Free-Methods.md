# Anchor-Free Methods

> **Simplifying the pipeline** - CenterNet and FCOS

---

## ⚓ The Problem with Anchors

Anchor-based methods (YOLOv3, Faster R-CNN) are annoying:
1.  **Hyperparameters:** Need to tune anchor sizes/ratios for every dataset.
2.  **Imbalance:** 100k anchors, only 50 objects.
3.  **Complexity:** IoU matching logic is complex.

---

## 🎯 CenterNet (Objects as Points)

**Idea:** Detect the **center point** of an object using a heatmap.
Then regress width and height at that center location.

**Output Heads:**
1.  **Heatmap ($C$ channels):** Peaks indicate object centers.
2.  **Offset (2 channels):** Refine discretization error.
3.  **Size (2 channels):** Width and Height.

**Pros:** No NMS needed! (MaxPooling on heatmap acts as NMS).

---

## 📍 FCOS (Fully Convolutional One-Stage)

**Idea:** Predict a 4D vector $(l, t, r, b)$ at every pixel.
- Distance to Left, Top, Right, Bottom of the bounding box.

**Center-ness:** A branch that predicts how close a pixel is to the center of the box. Down-weights low-quality predictions at the edges.

---

## 🏆 Detection Transformers (DETR)

**The Ultimate Anchor-Free:**
Uses a Transformer Encoder-Decoder.
Input: Image Features.
Output: Set of Box Predictions.
**Bipartite Matching:** Uses Hungarian Algorithm to match predictions to ground truth 1-to-1.
No Anchors. No NMS. Just pure set prediction.

---

## 🎓 Interview Focus

1.  **Why are Anchor-Free methods gaining popularity?**
    - Simpler design. No need to cluster dataset to find optimal anchor sizes.
    - YOLOv8 is anchor-free.

2.  **How does CenterNet handle overlapping objects?**
    - If two objects of the *same class* have the exact same center, it collides.
    - Rare in practice.

3.  **What is the "Centerness" branch in FCOS?**
    - Pixels near the edge of an object produce poor box regressions. Centerness suppresses these low-quality predictions during inference.

---

**Anchor-Free: Less heuristics, more learning!**
