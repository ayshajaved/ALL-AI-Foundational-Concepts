# Panoptic Segmentation

> **The Unified View** - Stuff vs Things

---

## 🎯 The Task

Combines **Semantic** and **Instance** segmentation into one unified output.

- **Things (Countable):** Person, Car, Cat. (Instance-level).
- **Stuff (Amorphous):** Sky, Road, Grass, Water. (Semantic-level).

**Goal:** Assign a unique `(Class ID, Instance ID)` to *every* pixel in the image.

---

## 🏗️ Panoptic FPN

A single network with a shared backbone (ResNet + FPN) and two heads:
1.  **Instance Head (Mask R-CNN):** Detects "Things".
2.  **Semantic Head (FCN):** Segments "Stuff".

**Post-Processing (Fusion):**
- If the Instance Head predicts a "Car" mask, those pixels are assigned to that car instance.
- Remaining pixels are filled by the Semantic Head (e.g., "Road").
- Resolves overlaps (e.g., Instance mask takes precedence over Semantic mask).

---

## 🏆 Panoptic Quality (PQ)

The metric for Panoptic Segmentation.
Combines **Segmentation Quality (SQ)** (IoU of matched segments) and **Recognition Quality (RQ)** (F1 score of detection).

$$ PQ = \underbrace{\frac{\sum_{(p, g) \in TP} \text{IoU}(p, g)}{|TP|}}_{\text{SQ}} \times \underbrace{\frac{|TP|}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}}_{\text{RQ}} $$

---

## 🎓 Interview Focus

1.  **Why distinguish "Stuff" vs "Things"?**
    - "Stuff" doesn't have instances. You can't count "sky". It's just a region.
    - "Things" are countable objects.

2.  **What happens if two instance masks overlap?**
    - In Panoptic Segmentation, every pixel must belong to *exactly one* segment.
    - Heuristic: The object with higher confidence score wins the pixels. Or the one closer to the camera (if depth is known).

---

**Panoptic: The complete scene understanding!**
