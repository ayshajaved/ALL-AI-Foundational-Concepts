# Vision Transformers (ViT)

> **Transformers conquering Computer Vision** - Treating images as words

---

## 🎯 The Core Idea

Standard CNNs use convolutions to capture local features.
**ViT** asks: Can we apply a standard Transformer directly to images?

**Challenge:** Images are 2D grids of pixels. Transformers expect 1D sequences of tokens.
**Solution:** Patchify.

---

## 🏗️ ViT Architecture

1.  **Patch Partitioning:** Split image into fixed-size patches (e.g., $16 \times 16$).
2.  **Linear Projection:** Flatten each patch and map to a vector (embedding).
    - Image $224 \times 224$ $\to$ $196$ patches of $16 \times 16$.
3.  **Position Embeddings:** Add learnable position vectors (since patches have spatial order).
4.  **Transformer Encoder:** Standard BERT-like encoder.
5.  **MLP Head:** Classification head attached to the special `[CLS]` token.

### Inductive Bias
- **CNNs:** Strong inductive bias (translation invariance, locality). Good for small data.
- **ViT:** Weak inductive bias (global attention). Needs **massive data** (JFT-300M) or strong regularization to beat CNNs.

---

## 💻 PyTorch Implementation (Patch Embedding)

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Conv2d with stride=patch_size does the flattening and projection efficiently
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x) # (B, E, H/P, W/P)
        x = x.flatten(2) # (B, E, N_patches)
        x = x.transpose(1, 2) # (B, N_patches, E)
        return x

# Usage
x = torch.randn(1, 3, 224, 224)
pe = PatchEmbedding()
out = pe(x)
print(f"Patches shape: {out.shape}") # (1, 196, 768)
```

---

## 🚀 Swin Transformer

**Problem with ViT:** Quadratic complexity $O(N^2)$ with image resolution. High res = slow.
**Swin Solution:** Hierarchical Transformer with **Shifted Windows**.
- Compute attention only within local windows (linear complexity).
- Shift windows between layers to allow cross-window connection.
- Becomes a general-purpose backbone (like ResNet) for detection/segmentation.

---

## 🎓 Interview Focus

1.  **Why does ViT need more data than ResNet?**
    - ViT lacks the inductive biases of locality and translation invariance that are "hard-coded" into CNNs. It has to learn these spatial relationships from scratch.

2.  **How does ViT handle different image sizes?**
    - It can handle them, but the position embeddings are usually fixed size. You often need to interpolate position embeddings to match the new number of patches.

3.  **Difference between ViT and Swin?**
    - ViT: Global attention (all patches attend to all).
    - Swin: Local window attention (efficient) + Hierarchical structure.

---

**ViT: Unifying Vision and Language architectures!**
