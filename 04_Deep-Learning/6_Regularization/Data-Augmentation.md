# Data Augmentation

> **Free data** - Expanding your dataset without collecting more samples

---

## 🖼️ Image Augmentation

Standard transforms to make model invariant to translation, rotation, scale, color.

```python
from torchvision import transforms

aug = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

---

## ✂️ Advanced Augmentation

### Mixup
Linear interpolation of two images and their labels.
$$ x' = \lambda x_i + (1-\lambda)x_j $$
$$ y' = \lambda y_i + (1-\lambda)y_j $$

**Effect:** Encourages linear behavior in-between classes.

### CutMix
Cut a patch from one image and paste it onto another.

### AutoAugment / RandAugment
Learn or randomly select the best sequence of augmentations.

### Mixup Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

def show_mixup(x1, x2, lam):
    """
    Visualizes Mixup of two images.
    """
    mixed_x = lam * x1 + (1 - lam) * x2
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(x1.permute(1, 2, 0))
    plt.title(f"Image 1")
    
    plt.subplot(1, 3, 2)
    plt.imshow(x2.permute(1, 2, 0))
    plt.title(f"Image 2")
    
    plt.subplot(1, 3, 3)
    plt.imshow(mixed_x.permute(1, 2, 0))
    plt.title(f"Mixup (λ={lam:.2f})")
    plt.show()
```

---

## 🛠️ Albumentations (Industry Standard)

While `torchvision` is great, **Albumentations** is the industry standard for performance and variety (especially for segmentation masks and bounding boxes).

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.RandomResizedCrop(height=224, width=224),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(p=0.5),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Usage
# augmented = transform(image=image)["image"]
```

---

## 📝 Text Augmentation

Harder than images (must preserve meaning).

1.  **Back Translation:** English $\to$ French $\to$ English.
2.  **Synonym Replacement:** Replace words with synonyms (WordNet).
3.  **Random Deletion/Swap:** Robustness to noise.

---

## 🎓 Interview Focus

1.  **Why use Data Augmentation?**
    - Reduces overfitting.
    - Improves generalization.
    - Teaches invariances (e.g., a cat is still a cat if rotated).

2.  **What is Test Time Augmentation (TTA)?**
    - Augmenting the test image multiple times (e.g., flips, crops), predicting on all, and averaging results. Boosts accuracy.

---

**Augmentation: The cheapest way to improve performance!**
