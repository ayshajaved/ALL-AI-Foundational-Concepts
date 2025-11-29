# MobileNet & Edge AI

> **AI on your phone** - Depthwise Separable Convolutions

---

## 📱 The Constraint

Mobile devices have limited Compute (CPU/NPU) and Battery.
Standard Convolutions are too heavy.

**Standard Conv Cost:** $D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F$
- $D_K$: Kernel Size
- $M$: Input Channels
- $N$: Output Channels
- $D_F$: Feature Map Size

---

## ✂️ Depthwise Separable Convolution

Splits convolution into two steps:

1.  **Depthwise Conv:** Apply a single filter per input channel. (Spatial features).
    - Cost: $D_K \cdot D_K \cdot M \cdot D_F \cdot D_F$
2.  **Pointwise Conv:** $1 \times 1$ Conv to combine channels. (Channel features).
    - Cost: $1 \cdot 1 \cdot M \cdot N \cdot D_F \cdot D_F$

**Reduction:** $\frac{1}{N} + \frac{1}{D_K^2}$.
For a $3 \times 3$ kernel, this is **8-9x less computation** with minimal accuracy drop.

---

## 🚀 MobileNet Evolution

- **V1:** Introduced Depthwise Separable Convs.
- **V2:** Introduced **Inverted Residuals** and **Linear Bottlenecks** (Remove ReLU in the output of the bottleneck to preserve info).
- **V3:** Used **Neural Architecture Search (NAS)** and **Hard-Swish** activation (faster to compute than Sigmoid).

---

## 💻 PyTorch Implementation

```python
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1. Depthwise (groups = in_channels)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                   padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
```

---

## 🎓 Interview Focus

1.  **What is the `groups` parameter in Conv2d?**
    - It controls connections between input and output channels.
    - `groups=1`: Standard Conv (All inputs connected to all outputs).
    - `groups=in_channels`: Depthwise Conv (Each input has its own filter).

2.  **Why remove ReLU in the bottleneck (MobileNetV2)?**
    - ReLU destroys information in low-dimensional manifolds (the bottleneck). Linear activation preserves it.

---

**MobileNet: Making AI ubiquitous!**
