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
