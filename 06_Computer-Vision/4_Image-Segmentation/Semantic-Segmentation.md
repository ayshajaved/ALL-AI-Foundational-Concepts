# Semantic Segmentation

> **Pixel-level Classification** - FCN and U-Net

---

## 🎯 The Task

**Classification:** "Cat" (1 label per image).
**Detection:** "Cat at [x,y,w,h]" (Box per object).
**Semantic Segmentation:** "Pixel (i, j) is Cat" (Label per pixel).

**Output:** A mask of shape $(H \times W)$ where each value is a class ID.
*Note:* Does NOT distinguish between two different cats. All cat pixels are just "Cat".

---

## 🏗️ Fully Convolutional Networks (FCN) - 2015

**Idea:** Take a classification network (e.g., VGG16), remove the Dense layers, and replace them with $1 \times 1$ convolutions.
**Upsampling:** Use **Transposed Convolutions** (Deconvolution) to scale the small feature map back up to the original image size.

**Problem:** Result is coarse/blurry because pooling layers destroyed spatial information.

---

## ∪ U-Net - 2015

**Architecture:** Encoder-Decoder with **Skip Connections**.

1.  **Encoder (Contracting Path):** Standard CNN (Conv + Pool). Captures **Context** ("What is it?").
2.  **Decoder (Expanding Path):** Upsampling + Conv. Captures **Localization** ("Where is it?").
3.  **Skip Connections:** Concatenate high-resolution features from the Encoder directly to the Decoder. Recovers fine details (edges) lost during pooling.

$$ \text{Input} \to \text{Encoder} \to \text{Bottleneck} \to \text{Decoder} \to \text{Output} $$

---

## 💻 PyTorch Implementation (U-Net)

```python
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part
        features = [64, 128, 256, 512]
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x) # Up
            skip_connection = skip_connections[idx//2]
            
            # Concatenate
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip) # Double Conv

        return self.final_conv(x)
```

---

## 🎓 Interview Focus

1.  **Why do we need Skip Connections in U-Net?**
    - Pooling layers lose spatial information (resolution). The decoder needs this info to draw sharp boundaries. Skip connections provide "high-res" features from the encoder.

2.  **Transposed Conv vs Bilinear Upsampling?**
    - **Transposed Conv:** Learnable upsampling. Can learn to fill gaps intelligently.
    - **Bilinear:** Fixed mathematical formula. Faster, no parameters.
    - Modern U-Nets often use Bilinear + Conv to avoid "Checkerboard Artifacts".

---

**Semantic Segmentation: Painting by numbers!**
