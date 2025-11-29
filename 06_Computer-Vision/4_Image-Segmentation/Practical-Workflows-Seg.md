# Practical Workflows: Background Remover

> **Building a Zoom Background Tool** - DeepLabV3+ Inference

---

## 🛠️ The Project

Remove the background from a selfie and replace it with a new image.
**Model:** DeepLabV3 (ResNet-101 backbone).

---

## 💻 Implementation

```python
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Model
model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

# 2. Preprocess
input_image = Image.open("selfie.jpg")
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image).unsqueeze(0)

# 3. Inference
with torch.no_grad():
    output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0) # [H, W] Class IDs

# 4. Create Mask
# Class 15 is 'person' in COCO/Pascal VOC
mask = output_predictions == 15 

# 5. Apply Mask
# Convert PIL to Numpy
img_np = np.array(input_image)
mask_np = mask.numpy()

# Create alpha channel
h, w, c = img_np.shape
result = np.zeros((h, w, 4), dtype=np.uint8)
result[:, :, :3] = img_np
result[:, :, 3] = mask_np * 255 # Alpha: 255 for person, 0 for background

# Save
Image.fromarray(result).save("transparent_selfie.png")
```

---

## 🧠 Optimizations for Real-Time (Webcam)

1.  **Model Selection:** Use `deeplabv3_mobilenet_v3_large` instead of ResNet-101.
2.  **Resolution:** Run inference at low res ($256 \times 256$), then upsample the mask to HD.
3.  **Edge Filtering:** Use a "Guided Filter" to refine the jagged edges of the low-res mask using the high-res image as a guide.

---

**Segmentation: The green screen of the future!**
