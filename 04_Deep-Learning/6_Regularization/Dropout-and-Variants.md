# Dropout and Variants

> **Preventing co-adaptation of neurons** - The most popular regularization technique

---

## 🎯 Dropout

**Idea:** Randomly zero out neurons during training with probability $p$.
**Effect:** Forces network to learn robust features that don't rely on specific neurons. Ensemble effect (training $2^N$ sub-networks).

**Inference:** Multiply weights by $(1-p)$ (or scale by $1/(1-p)$ during training, called Inverted Dropout).

```python
import torch.nn as nn

# p=0.5 probability of zeroing out
dropout = nn.Dropout(p=0.5)

x = torch.randn(1, 10)
out = dropout(x)
```

---

## 📉 Spatial Dropout (Dropout2d)

**Problem:** Standard dropout on CNN feature maps drops individual pixels. Pixels are highly correlated, so information leaks.
**Solution:** Drop entire **channels** (feature maps).

```python
# Drops entire channels
spatial_dropout = nn.Dropout2d(p=0.5)

x = torch.randn(1, 32, 10, 10) # (Batch, Channel, H, W)
out = spatial_dropout(x)
```

---

## 🎲 DropConnect

**Idea:** Instead of dropping activations (neurons), drop **weights** (connections).
**Result:** Generalization of Dropout.

---

## 📉 Stochastic Depth

**Used in:** ResNets, Vision Transformers.
**Idea:** Randomly drop entire **layers** (residual blocks) during training.
**Effect:** Effectively trains shallower networks that act as deep networks during inference.

```python
from torchvision.ops import StochasticDepth

stochastic_depth = StochasticDepth(p=0.2, mode="row")
```

---

## 🎓 Interview Focus

1.  **Why does Dropout work?**
    - Prevents overfitting by preventing neurons from co-adapting.
    - Acts as an approximate model averaging (ensemble) technique.

2.  **Dropout during Test time?**
    - Usually turned OFF.
    - If kept ON (Monte Carlo Dropout), it estimates model uncertainty.

3.  **Spatial Dropout vs Standard Dropout?**
    - Spatial drops entire channels (good for CNNs). Standard drops independent elements.

---

**Dropout: The "delete" key for overfitting!**
