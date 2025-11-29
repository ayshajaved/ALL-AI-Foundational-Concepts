# Geometric Transformations

> **Warping Space** - Rotation, Affine, and Perspective Transforms

---

## 🔄 Basic Transformations

### 1. Rotation
Need a rotation matrix.
```python
rows, cols = img.shape[:2]
# Center, Angle, Scale
M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1) 
rotated = cv2.warpAffine(img, M, (cols, rows))
```

### 2. Translation (Shifting)
Matrix: $\begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \end{bmatrix}$
```python
M = np.float32([[1, 0, 100], [0, 1, 50]]) # Shift x+100, y+50
shifted = cv2.warpAffine(img, M, (cols, rows))
```

---

## 📐 Affine Transformation

**Properties:** Parallel lines remain parallel.
Needs **3 points** in input and output to define.
(e.g., Skewing, correcting slight tilts).

```python
pts1 = np.float32([[50,50], [200,50], [50,200]])
pts2 = np.float32([[10,100], [200,50], [100,250]])

M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols, rows))
```

---

## 🏙️ Perspective Transformation (Homography)

**Properties:** Parallel lines converge (like train tracks).
Needs **4 points** to define.
**Use Case:** Document scanning (correcting a slanted paper to be flat).

```python
pts1 = np.float32([[56,65], [368,52], [28,387], [389,390]]) # Corners of paper
pts2 = np.float32([[0,0], [300,0], [0,300], [300,300]])     # Flat square

M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img, M, (300, 300))
```

---

## 🎓 Interview Focus

1.  **Affine vs Perspective Transform?**
    - **Affine:** Preserves parallelism. 3 points. (2x3 Matrix).
    - **Perspective:** Does NOT preserve parallelism. 4 points. (3x3 Matrix).

2.  **What is Homography?**
    - A mapping between two planar surfaces in space. Used in Panorama stitching to align images.

3.  **How to find the transformation matrix automatically?**
    - Use Feature Matching (SIFT/ORB) to find matching points between two images, then use `cv2.findHomography`.

---

**Geometry: Correcting your camera's perspective!**
