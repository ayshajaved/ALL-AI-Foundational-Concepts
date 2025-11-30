# Video Data Processing

> **The Fourth Dimension** - Handling Spatiotemporal Data

---

## 🎞️ Video as a Volume

An image is $H \times W \times C$.
A video is $T \times H \times W \times C$.
- $T$: Time (Number of frames).
- **Size Explosion:** A 10-second video at 30fps ($224 \times 224$) is $300 \times 224 \times 224 \times 3 \approx 45$ million values.

---

## 🧊 3D Convolutions (C3D / I3D)

Standard 2D CNNs process frames independently (ignoring motion).
**3D CNNs** use kernels of size $k_t \times k_h \times k_w$ (e.g., $3 \times 3 \times 3$).
- They slide across Time, Height, and Width.
- They capture **Motion** and **Appearance** simultaneously.

**I3D (Inflated 3D ConvNet):**
- Take a pre-trained 2D ResNet.
- "Inflate" the $k \times k$ filters to $k \times k \times k$ by copying weights across the time dimension.
- Allows using ImageNet weights for Video.

---

## ✂️ Sampling Strategies

We cannot feed the whole video to the GPU.
1.  **Uniform Sampling:** Pick 16 frames evenly spaced.
2.  **Dense Sampling:** Pick a clip of 16 consecutive frames (for fine motion).
3.  **Stride:** Pick every $k$-th frame.

---

## 💻 PyTorch Video (Torchvision)

```python
import torch
import torchvision
from torchvision.io import read_video

# 1. Read Video
# Returns: [Time, Height, Width, Channels]
frames, audio, metadata = read_video("video.mp4", pts_unit="sec")

# 2. Permute to [Channels, Time, Height, Width] for 3D CNN
frames = frames.permute(3, 0, 1, 2).float() / 255.0

# 3. 3D Convolution
conv3d = torch.nn.Conv3d(
    in_channels=3, 
    out_channels=64, 
    kernel_size=(3, 3, 3), 
    padding=1
)

output = conv3d(frames.unsqueeze(0)) # Add Batch Dim
print(output.shape)
```

---

## 🎓 Interview Focus

1.  **Why is 3D Conv expensive?**
    - Parameters increase by factor $k_t$.
    - Compute increases by factor $T$.
    - Memory usage is the bottleneck.

2.  **Two-Stream Networks?**
    - An alternative to 3D CNNs.
    - Stream 1: Spatial (RGB frame).
    - Stream 2: Temporal (Optical Flow).
    - Fuse predictions at the end.

---

**Video Processing: Adding Time to Space!**
