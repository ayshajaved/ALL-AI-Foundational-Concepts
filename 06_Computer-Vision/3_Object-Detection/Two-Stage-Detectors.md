# Two-Stage Detectors

> **Accuracy First** - R-CNN, Fast R-CNN, and Faster R-CNN

---

## 🐢 R-CNN (Regions with CNN) - 2014

**Pipeline:**
1.  **Region Proposals:** Use "Selective Search" (CPU algorithm) to find ~2000 candidate boxes.
2.  **Warp:** Resize every box to $224 \times 224$.
3.  **CNN:** Run AlexNet on *every* box independently.
4.  **Classify:** SVM to classify features.
5.  **Regress:** Refine box coordinates.

**Problem:** Slow! 2000 forward passes per image. (47s per image).

---

## 🐇 Fast R-CNN - 2015

**Innovation:** Run CNN on the **whole image** once.
1.  **Feature Map:** Generate features for the whole image.
2.  **ROI Pooling:** Project the region proposals onto the feature map and extract fixed-size features.
3.  **FC Layers:** Classify and regress.

**Speed:** 100x faster than R-CNN.
**Bottleneck:** Region Proposal (Selective Search) is still slow.

---

## 🚀 Faster R-CNN - 2015

**Innovation:** **Region Proposal Network (RPN)**.
Replaced Selective Search with a neural network.
The RPN slides over the feature map and predicts:
1.  **Objectness Score:** Is there an object here?
2.  **Box Offsets:** Adjust the anchor box.

**Pipeline:**
1.  **Backbone (ResNet):** Extract features.
2.  **RPN:** Propose regions.
3.  **ROI Align (Mask R-CNN improvement):** Extract features for proposals.
4.  **Head:** Classify and Refine.

**Result:** Real-time(ish). 5-10 FPS. SOTA Accuracy.

---

## 💻 PyTorch Implementation

```python
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# 1. Load Pre-trained
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 2. Inference
import torch
from PIL import Image
from torchvision import transforms

img = Image.open("street.jpg")
transform = transforms.Compose([transforms.ToTensor()])
img_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    predictions = model(img_tensor)

# predictions[0]['boxes']
# predictions[0]['labels']
# predictions[0]['scores']
```

---

## 🎓 Interview Focus

1.  **ROI Pooling vs ROI Align?**
    - **ROI Pooling:** Quantizes coordinates (snaps to grid). Loses spatial precision. Bad for segmentation.
    - **ROI Align:** Uses Bilinear Interpolation. Preserves exact spatial alignment. Crucial for Mask R-CNN.

2.  **Why is Faster R-CNN "Two-Stage"?**
    - Stage 1: Propose regions (RPN).
    - Stage 2: Classify regions.
    - One-Stage detectors (YOLO) do both in a single pass.

---

**Faster R-CNN: The grandfather of modern detection!**
