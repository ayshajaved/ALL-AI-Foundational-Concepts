# Advanced CNN Techniques

> **Modern CNN innovations** - Depthwise separable, dilated, and more

---

## 🎯 Depthwise Separable Convolutions

**Idea:** Factorize standard convolution into depthwise + pointwise

```python
# Standard conv
standard = nn.Conv2d(64, 128, 3, padding=1)
# Params: 3×3×64×128 = 73,728

# Depthwise separable
depthwise = nn.Conv2d(64, 64, 3, padding=1, groups=64)
pointwise = nn.Conv2d(64, 128, 1)
# Params: (3×3×64) + (64×128) = 8,768

separable = nn.Sequential(depthwise, pointwise)
```

**Used in:** MobileNet, EfficientNet

---

## 📊 Dilated Convolutions

**Idea:** Increase receptive field without increasing parameters

```python
# Dilation rate = 2
dilated_conv = nn.Conv2d(64, 64, 3, padding=2, dilation=2)

# Receptive field increases without pooling
```

**Used in:** Semantic segmentation, audio processing

---

## 🎯 1×1 Convolutions

**Uses:**
- Dimensionality reduction
- Channel mixing
- Adding non-linearity

```python
# Reduce channels
reduce = nn.Conv2d(256, 64, 1)

input = torch.randn(1, 256, 28, 28)
output = reduce(input)  # (1, 64, 28, 28)
```

---

## 📈 Squeeze-and-Excitation (SE) Blocks

**Idea:** Channel attention mechanism

```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.squeeze(x).view(b, c)
        # Excitation
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale
        return x * y
```

---

**Advanced techniques: pushing CNN performance!**
