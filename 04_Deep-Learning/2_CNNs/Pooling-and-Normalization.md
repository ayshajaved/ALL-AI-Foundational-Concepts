# Pooling and Normalization

> **Reducing dimensions and stabilizing training** - Essential CNN components

---

## 🎯 Pooling Layers

### Max Pooling
```python
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

# Reduces spatial dimensions by half
input = torch.randn(1, 64, 28, 28)
output = max_pool(input)  # (1, 64, 14, 14)
```

### Average Pooling
```python
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
```

### Global Average Pooling
```python
global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

input = torch.randn(1, 512, 7, 7)
output = global_avg_pool(input)  # (1, 512, 1, 1)
```

---

## 📊 Batch Normalization

```
BN(x) = γ((x - μ)/σ) + β

μ, σ: batch statistics
γ, β: learnable parameters
```

```python
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
```

**Benefits:**
- Faster training
- Higher learning rates
- Regularization effect

---

## 🎯 Layer Normalization

```python
layer_norm = nn.LayerNorm([64, 28, 28])
```

**Use:** Transformers, RNNs

---

## 📈 Group Normalization

```python
group_norm = nn.GroupNorm(num_groups=32, num_channels=64)
```

**Advantage:** Works with small batch sizes

---

**Pooling and normalization: essential CNN building blocks!**
