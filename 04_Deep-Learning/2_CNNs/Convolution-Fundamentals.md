# Convolution Fundamentals

> **The core operation of CNNs** - Understanding convolution, filters, and feature maps

---

## 🎯 Convolution Operation

### 2D Convolution
```
Output[i,j] = Σₘ Σₙ Input[i+m, j+n] × Kernel[m,n]
```

### Example

```python
import torch
import torch.nn as nn

# Input: (batch, channels, height, width)
input = torch.randn(1, 1, 5, 5)

# Conv layer: in_channels=1, out_channels=1, kernel_size=3
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)

output = conv(input)
print(f"Input shape: {input.shape}")   # (1, 1, 5, 5)
print(f"Output shape: {output.shape}") # (1, 1, 3, 3)
```

---

## 📊 Key Concepts

### Stride
```
stride=1: slide 1 pixel at a time
stride=2: slide 2 pixels at a time
```

```python
conv_stride2 = nn.Conv2d(1, 1, kernel_size=3, stride=2)
output = conv_stride2(input)
print(f"Output shape: {output.shape}") # (1, 1, 2, 2)
```

### Padding
```
padding='same': output size = input size
padding='valid': no padding
```

```python
conv_padded = nn.Conv2d(1, 1, kernel_size=3, padding=1)
output = conv_padded(input)
print(f"Output shape: {output.shape}") # (1, 1, 5, 5)
```

### Output Size Formula
```
Output_size = ⌊(Input_size + 2×Padding - Kernel_size)/Stride⌋ + 1
```

---

## 🎯 Receptive Field

**Definition:** Region of input that affects a particular output

```python
# Stack convolutions
model = nn.Sequential(
    nn.Conv2d(1, 16, 3),  # RF: 3×3
    nn.Conv2d(16, 32, 3), # RF: 5×5
    nn.Conv2d(32, 64, 3)  # RF: 7×7
)
```

---

## 📈 Parameter Sharing

**Advantage:** Fewer parameters than fully connected

```
Conv params: kernel_size² × in_channels × out_channels + out_channels (bias)
FC params: input_size × output_size + output_size
```

---

**Convolution: the foundation of computer vision!**
