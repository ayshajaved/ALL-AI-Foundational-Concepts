# Video Diffusion Models (VDM)

> **Extending the Magic** - From Images to Video

---

## 🌌 The Concept

Standard Diffusion Models (DDPM/LDM) learn to denoise 2D images.
**Video Diffusion Models** learn to denoise 3D spatiotemporal volumes ($T \times H \times W \times C$).

$$ p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)) $$

Where $x$ is now a video clip.

---

## 🏗️ Architecture: 3D U-Net

The backbone is still a U-Net, but modified for 3D.
1.  **Spatial Layers:** 2D Convolutions process each frame independently (sharing weights).
2.  **Temporal Layers:** 1D Convolutions or Attention blocks process the time dimension (mixing information across frames).
    - **Factorized Attention:** Instead of full $Attention(T \times H \times W)$, we do:
        - Spatial Attention: $Attention(H \times W)$ for each $t$.
        - Temporal Attention: $Attention(T)$ for each $(h, w)$.

---

## 🎓 Joint Training (Image + Video)

Video data is scarce and expensive. Image data is abundant.
VDMs are often trained jointly:
- **Image Batch:** Treat as video with $T=1$. Skip temporal layers.
- **Video Batch:** Use full model.
- This allows the model to learn high-quality textures from images and motion dynamics from video.

---

## 💻 PyTorch Concept (Temporal Attention)

```python
import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads)
        
    def forward(self, x):
        # x shape: [Batch, Channels, Time, Height, Width]
        b, c, t, h, w = x.shape
        
        # Reshape for Attention: [Time, Batch*Height*Width, Channels]
        x = x.permute(2, 0, 3, 4, 1).reshape(t, b*h*w, c)
        
        # Self-Attention over Time
        x, _ = self.attn(x, x, x)
        
        # Reshape back
        x = x.reshape(t, b, h, w, c).permute(1, 4, 0, 2, 3)
        return x
```

---

## 🎓 Interview Focus

1.  **Why Factorized Attention?**
    - Full attention is $O((THW)^2)$.
    - Factorized is $O(T(HW)^2 + HW(T)^2)$. Much more efficient.

2.  **What is "Ho Video"?**
    - A seminal paper (Video Diffusion Models, 2022) that showed standard 2D U-Nets can be easily adapted to video by interleaving temporal attention layers.

---

**VDM: The foundation of Sora and Runway Gen-2!**
