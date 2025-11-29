# Image Filtering

> **Kernels and Convolutions** - Blurring, Sharpening, and Edge Detection

---

## 🧠 The Concept: Convolution

A **Kernel** (small matrix, e.g., $3 \times 3$) slides over the image.
At each position, we do an element-wise multiplication and sum.

$$ Output(x,y) = \sum \sum I(x-i, y-j) \cdot K(i,j) $$

---

## 🌫️ Blurring (Smoothing)

Used to remove noise.

### 1. Gaussian Blur
Kernel values follow a bell curve. Center pixel matters most.
```python
# (5, 5) is kernel size, 0 is sigmaX (auto-calculated)
blur = cv2.GaussianBlur(img, (5, 5), 0)
```

### 2. Median Blur
Replaces pixel with median of neighbors.
**Superpower:** Removes "Salt and Pepper" noise perfectly while preserving edges.
```python
median = cv2.medianBlur(img, 5)
```

---

## 🔪 Edge Detection

Finding boundaries where intensity changes drastically.

### 1. Sobel Operator
Calculates gradient (derivative) in X and Y directions.
- $G_x$: Vertical edges.
- $G_y$: Horizontal edges.

```python
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
```

### 2. Canny Edge Detector (The Gold Standard)
Multi-stage algorithm:
1.  Gaussian Blur (Noise reduction).
2.  Sobel Gradient calculation.
3.  **Non-Maximum Suppression:** Thin out edges (keep only the peak).
4.  **Hysteresis Thresholding:** Use two thresholds (High/Low) to link weak edges to strong edges.

```python
# 100 = Low Threshold, 200 = High Threshold
edges = cv2.Canny(img, 100, 200)
```

---

## 🎓 Interview Focus

1.  **Gaussian vs Median Blur?**
    - **Gaussian:** Good for random noise. Blurs edges slightly.
    - **Median:** Excellent for Salt-and-Pepper noise. Preserves edges better.

2.  **Why does Canny use two thresholds?**
    - To solve the trade-off between missing edges (threshold too high) and detecting noise (threshold too low).
    - Strong edges (> High) are kept. Weak edges (between Low and High) are kept *only if* connected to a strong edge.

3.  **What is a Laplacian filter?**
    - Second derivative. Detects regions of rapid intensity change (edges) regardless of orientation. Very sensitive to noise.

---

**Filtering: Manipulating reality pixel by pixel!**
