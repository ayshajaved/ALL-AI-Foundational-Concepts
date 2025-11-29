# Image Basics

> **Pixels, Channels, and Color Spaces** - The atoms of Computer Vision

---

## 🖼️ What is an Image?

To a computer, an image is just a **multidimensional array (tensor)** of numbers.
- **Grayscale:** 2D array $(H \times W)$. Values $0$ (Black) to $255$ (White).
- **Color (RGB):** 3D array $(H \times W \times 3)$. Three channels: Red, Green, Blue.

$$ I(x, y) = [R, G, B] $$

---

## 🎨 Color Spaces

Different ways to represent color for different tasks.

### 1. RGB (Red, Green, Blue)
- **Standard:** Used by cameras and screens.
- **Problem:** Correlated channels. Changing brightness affects R, G, and B. Hard to separate "color" from "intensity".

### 2. HSV (Hue, Saturation, Value)
- **Hue:** The color type ($0-360^\circ$). Red, Blue, Yellow.
- **Saturation:** Intensity of color (Gray $\to$ Vivid).
- **Value:** Brightness (Dark $\to$ Light).
- **Use Case:** Color-based object tracking (e.g., "Track the red ball"). We can threshold Hue while ignoring lighting changes (Value).

### 3. LAB (Lightness, A, B)
- **L:** Lightness (0-100).
- **A:** Green $\leftrightarrow$ Red.
- **B:** Blue $\leftrightarrow$ Yellow.
- **Use Case:** Perceptually uniform. The Euclidean distance between two colors in LAB space matches human perception difference.

---

## 💻 Python Implementation (OpenCV & NumPy)

OpenCV loads images in **BGR** format by default (historical quirk).

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Image
# 'image.jpg' is loaded as BGR
img_bgr = cv2.imread('image.jpg')

# 2. Convert to RGB (for Matplotlib)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 3. Convert to Grayscale
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# 4. Convert to HSV
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# Visualize
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1); plt.imshow(img_rgb); plt.title("RGB")
plt.subplot(1, 2, 2); plt.imshow(img_gray, cmap='gray'); plt.title("Grayscale")
plt.show()
```

---

## 🎓 Interview Focus

1.  **Why BGR instead of RGB in OpenCV?**
    - Historical reasons. Early camera manufacturers and software used BGR. OpenCV kept it for backward compatibility.

2.  **When to use HSV over RGB?**
    - When you need to detect objects based on color under varying lighting conditions. In RGB, a shadow changes all three values. In HSV, it mostly changes only 'V'.

3.  **What is the shape of a 1080p RGB image?**
    - $(1080, 1920, 3)$.
    - Memory: $1080 \times 1920 \times 3$ bytes $\approx 6.2$ MB (uint8).

---

**Basics: Seeing the matrix!**
