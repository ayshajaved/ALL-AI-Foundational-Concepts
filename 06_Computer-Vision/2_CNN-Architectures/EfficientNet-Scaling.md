# EfficientNet Scaling

> **Smarter, not just bigger** - Compound Scaling

---

## 🎯 The Scaling Dilemma

To improve accuracy, we can scale:
1.  **Depth ($\alpha$):** More layers (ResNet-152). Captures complex features. Harder to train.
2.  **Width ($\beta$):** More channels (WideResNet). Captures fine-grained features. Expensive.
3.  **Resolution ($\gamma$):** Larger images ($512 \times 512$). More detail. Slow.

**Problem:** Scaling just one dimension hits diminishing returns.

---

## ⚖️ Compound Scaling (EfficientNet)

**Idea:** Scale all three dimensions uniformly using a compound coefficient $\phi$.

$$ \text{Depth: } d = \alpha^\phi $$
$$ \text{Width: } w = \beta^\phi $$
$$ \text{Resolution: } r = \gamma^\phi $$

Subject to: $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ (Doubles FLOPS).

**Result:** EfficientNet-B7 achieves State-of-the-Art (SOTA) accuracy with **8.4x fewer parameters** and **6.1x faster inference** than GPipe.

---

## 🧱 MBConv (Inverted Residual Block)

EfficientNet uses the **Mobile Inverted Bottleneck Convolution** (from MobileNetV2).
1.  **Expansion ($1 \times 1$):** Expand channels (Low $\to$ High).
2.  **Depthwise Conv ($3 \times 3$):** Spatial filtering.
3.  **Squeeze-and-Excitation (SE):** Attention mechanism to weight channels.
4.  **Projection ($1 \times 1$):** Project back (High $\to$ Low).
5.  **Residual Connection:** Add input to output.

---

## 💻 Using EfficientNet (PyTorch)

```python
from torchvision import models

# Load Pre-trained
model = models.efficientnet_b0(pretrained=True)

# Inspect structure
print(model.features[0]) 
# Conv2dNormActivation(...)
```

---

## 🎓 Interview Focus

1.  **Why "Inverted" Residuals?**
    - Standard ResNet: Wide $\to$ Narrow $\to$ Wide.
    - Inverted (MobileNet): Narrow $\to$ Wide $\to$ Narrow.
    - This preserves information in the "Wide" intermediate layers while keeping the input/output tensors small (memory efficient).

2.  **What is Squeeze-and-Excitation (SE)?**
    - A mini-network that looks at global information (Global Avg Pool) and outputs a weight vector (0 to 1) to scale each channel. "Pay attention to the 'Dog' channel, ignore the 'Background' channel."

---

**EfficientNet: The gold standard for efficiency!**
