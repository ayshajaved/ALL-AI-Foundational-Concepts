# Instance Segmentation

> **Detecting individuals** - Mask R-CNN

---

## 🎯 The Task

**Semantic:** "These pixels are Person."
**Instance:** "These pixels are Person A. Those pixels are Person B."

Combines **Object Detection** (Box) + **Semantic Segmentation** (Mask).

---

## 🎭 Mask R-CNN (2017)

Extends **Faster R-CNN** by adding a third branch for predicting the mask.

**Outputs per candidate object:**
1.  **Class Label:** "Person"
2.  **Bounding Box:** `[x, y, w, h]`
3.  **Mask:** Binary mask $(28 \times 28)$ inside the box.

---

## 🔑 Key Innovation: RoI Align

**Problem with RoI Pooling (Faster R-CNN):**
It snaps coordinates to the nearest integer (quantization).
If a box is at `x=10.5`, RoI Pooling rounds to `10`.
For classification, this is fine. For pixel-perfect masks, this causes misalignment.

**Solution: RoI Align**
Uses **Bilinear Interpolation** to compute feature values at exact floating-point coordinates (`10.5`).
No quantization. Result: Massive improvement in mask accuracy.

---

## 💻 PyTorch Implementation

```python
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn

# 1. Load Pre-trained (COCO)
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 2. Inference
import torch
from PIL import Image
from torchvision import transforms

img = Image.open("street.jpg")
transform = transforms.Compose([transforms.ToTensor()])
x = transform(img).unsqueeze(0)

with torch.no_grad():
    prediction = model(x)

# prediction[0]['masks'] -> [N, 1, H, W] (Soft masks 0-1)
# prediction[0]['boxes']
# prediction[0]['labels']
```

---

## 🎓 Interview Focus

1.  **Semantic vs Instance Segmentation?**
    - Semantic treats all instances of a class as one blob. Instance separates them.
    - *Example:* A crowd of people. Semantic = One big "Person" blob. Instance = 50 individual "Person" blobs.

2.  **Why predict a $28 \times 28$ mask?**
    - Predicting a full-image mask ($1024 \times 1024$) for every object is too memory intensive.
    - We predict a small mask relative to the bounding box, then resize it back to the object's actual size on the image.

3.  **Can YOLO do Instance Segmentation?**
    - Yes! **YOLOv8-seg** adds a mask head (Proto-mask branch) to perform real-time instance segmentation.

---

**Mask R-CNN: The standard for instance-level understanding!**
