# OpenCV Essentials

> **The Swiss Army Knife of CV** - Reading, Writing, and Drawing

---

## 🛠️ Core Operations

### 1. I/O Operations

```python
import cv2

# Read
img = cv2.imread('input.jpg')

# Check if loaded
if img is None:
    print("Error loading image")

# Write
cv2.imwrite('output.png', img)
```

### 2. Resizing & Cropping

```python
# Resize to fixed dimensions
resized = cv2.resize(img, (300, 200)) # (Width, Height)

# Resize by ratio
scaled = cv2.resize(img, None, fx=0.5, fy=0.5)

# Crop (NumPy Slicing)
# img[y:y+h, x:x+w]
crop = img[50:150, 100:300]
```

### 3. Drawing Shapes

Useful for visualizing detection boxes.

```python
# Line
cv2.line(img, (0,0), (100,100), (255,0,0), 5) # Blue line

# Rectangle (Top-Left, Bottom-Right)
cv2.rectangle(img, (50,50), (200,200), (0,255,0), 3) # Green box

# Circle (Center, Radius)
cv2.circle(img, (150,150), 40, (0,0,255), -1) # Red filled circle

# Text
cv2.putText(img, "OpenCV", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
```

---

## 🔳 Thresholding (Binarization)

Converting a grayscale image to pure Black & White (0 or 255).
Essential for separating objects from background.

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Simple Thresholding
# If pixel > 127, set to 255, else 0
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Adaptive Thresholding (Better for varying lighting)
# Calculates threshold for small regions
adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)
```

---

## ➰ Contours

Finding boundaries of objects.

```python
# Find Contours
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw Contours
cv2.drawContours(img, contours, -1, (0,255,0), 3)

print(f"Found {len(contours)} objects")
```

---

## 🎓 Interview Focus

1.  **Interpolation methods in `resize`?**
    - `cv2.INTER_NEAREST`: Fastest, blocky.
    - `cv2.INTER_LINEAR`: Default, good for zooming.
    - `cv2.INTER_CUBIC`: Slowest, best quality.

2.  **Why use Adaptive Thresholding?**
    - Global thresholding fails if one part of the image is shadowed and another is bright. Adaptive calculates local thresholds.

3.  **What is `CHAIN_APPROX_SIMPLE`?**
    - Compresses horizontal, vertical, and diagonal segments. Instead of storing all points of a straight line, it only stores the two endpoints. Saves memory.

---

**OpenCV: The library that runs the world's cameras!**
