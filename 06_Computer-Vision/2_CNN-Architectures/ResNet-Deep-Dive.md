# ResNet Deep Dive

> **Solving the Vanishing Gradient** - Residual Connections and Bottlenecks

---

## 📉 The Degradation Problem

Ideally, adding more layers should decrease error.
In practice, plain deep networks (e.g., VGG-19) performed *worse* than shallower ones.
**Reason:** Not overfitting, but optimization difficulty. Gradients vanish through many layers.

---

## 🔗 The Residual Block

**Idea:** Instead of learning the mapping $H(x)$, learn the residual $F(x) = H(x) - x$.
Then $H(x) = F(x) + x$.
If the identity mapping is optimal, the weights drive $F(x) \to 0$.

$$ y = \text{ReLU}(F(x) + x) $$

**Skip Connection:** The signal $x$ bypasses the layers and is added to the output. This creates a "gradient superhighway" during backpropagation.

---

## 🍾 The Bottleneck Block (ResNet-50+)

For deeper networks, $3 \times 3$ convolutions are expensive.
**Solution:** $1 \times 1$ convolutions to reduce dimensions.

1.  **$1 \times 1$ Conv:** Reduce channels (e.g., $256 \to 64$).
2.  **$3 \times 3$ Conv:** Process features ($64 \to 64$).
3.  **$1 \times 1$ Conv:** Restore channels ($64 \to 256$).

This reduces parameters while maintaining depth.

---

## 💻 PyTorch Implementation

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # The magic addition
        out = self.relu(out)
        return out
```

---

## 🎓 Interview Focus

1.  **Why does ResNet work?**
    - Skip connections allow gradients to flow backwards without diminishing.
    - It acts like an ensemble of shallower networks (paths of varying lengths).

2.  **Why use BatchNorm?**
    - It normalizes layer inputs, reducing "Internal Covariate Shift". Allows higher learning rates and faster convergence.

3.  **ResNet-18 vs ResNet-50?**
    - **18/34:** Use Basic Blocks (2 layers).
    - **50/101/152:** Use Bottleneck Blocks (3 layers).

---

**ResNet: The backbone of modern vision!**
