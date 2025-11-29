# Practical Workflows: Document Scanner

> **Building a CamScanner Clone** - From messy photo to scanned PDF

---

## 🛠️ The Pipeline

1.  **Edge Detection:** Find the edges of the paper.
2.  **Contour Finding:** Find the largest rectangular contour.
3.  **Perspective Transform:** Warps the paper to a flat top-down view.
4.  **Thresholding:** Binarize to make it look like a scan.

---

## 💻 Implementation

```python
import cv2
import numpy as np

def order_points(pts):
    # Order points: TL, TR, BR, BL
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # TL
    rect[2] = pts[np.argmax(s)] # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # TR
    rect[3] = pts[np.argmax(diff)] # BL
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Compute width/height of new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
        
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# 1. Load & Preprocess
image = cv2.imread("receipt.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

# 2. Find Contours
cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

screenCnt = None
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    
    # If contour has 4 points, assume it's the paper
    if len(approx) == 4:
        screenCnt = approx
        break

if screenCnt is not None:
    # 3. Perspective Transform
    warped = four_point_transform(image, screenCnt.reshape(4, 2))
    
    # 4. Post-process (Scan effect)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    scanned = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
    
    cv2.imwrite("scanned_doc.jpg", scanned)
    print("Scan saved!")
else:
    print("No document found.")
```

---

## 🧠 Key Tricks

1.  **`approxPolyDP`:** Simplifies a jagged contour into a polygon. Crucial for finding the 4 corners of the paper.
2.  **Ordering Points:** The contour points come in random order. We must sort them (Top-Left, Top-Right...) to map them correctly to the destination rectangle.

---

**You just built a product used by millions!**
