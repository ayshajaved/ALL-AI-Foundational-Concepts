# Medical Imaging Segmentation

> **High Stakes AI** - 3D U-Net and Dice Loss

---

## 🏥 The Domain

Medical images (CT, MRI) are different:
1.  **3D Volumetric Data:** $(D \times H \times W)$.
2.  **Class Imbalance:** Tumor is 0.1% of the pixels. Background is 99.9%.
3.  **Privacy:** DICOM format, HIPAA compliance.

---

## 🏗️ 3D U-Net

Extension of U-Net to 3D.
Replaces 2D Convs with 3D Convs (`nn.Conv3d`).
- **Input:** $(1, Depth, Height, Width)$.
- **Kernels:** $3 \times 3 \times 3$.
- **Computationally Expensive:** Requires massive VRAM. Often trained on patches (sub-volumes).

---

## 📉 Dice Loss (The Savior)

Cross Entropy Loss fails when classes are imbalanced (model just predicts "Background" everywhere and gets 99.9% accuracy).

**Dice Coefficient (F1 Score for pixels):**
$$ D = \frac{2 |X \cap Y|}{|X| + |Y|} $$
- $X$: Predicted Mask.
- $Y$: Ground Truth Mask.

**Dice Loss:** $1 - D$.
Optimizes for **overlap**, not just pixel accuracy.

```python
def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()
```

---

## 🎓 Interview Focus

1.  **Why not use Cross Entropy for tumors?**
    - The gradients from the massive background dominate the gradients from the tiny tumor. The model learns to ignore the tumor.

2.  **What is V-Net?**
    - A variant of U-Net specifically designed for 3D medical segmentation, introducing residual connections in the blocks and using Dice Loss.

3.  **How to handle different MRI resolutions?**
    - **Resampling:** Interpolate all scans to a fixed spacing (e.g., 1mm $\times$ 1mm $\times$ 1mm) during preprocessing.

---

**Medical AI: Where precision saves lives!**
