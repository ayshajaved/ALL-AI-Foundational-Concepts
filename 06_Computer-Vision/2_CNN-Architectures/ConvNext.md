# ConvNeXt

> **A ConvNet for the 2020s** - Competing with Vision Transformers

---

## 🦖 The Rise of ViT

In 2020, Vision Transformers (ViT) overtook CNNs.
They had:
- Large receptive fields (Global attention).
- Modern training recipes (AdamW, Mixup).
- Macro design changes (Patchify, LayerNorm).

---

## 🧬 Modernizing the ResNet (ConvNeXt)

ConvNeXt (2022) took a standard ResNet-50 and applied "Transformer" design principles **without using Attention**.

**The Changes:**
1.  **Patchify:** Replaced the initial $7 \times 7$ stem with a $4 \times 4$ stride-4 conv (Non-overlapping patches).
2.  **Large Kernels:** Used $7 \times 7$ depthwise convs (up from $3 \times 3$). Mimics the global reach of Attention.
3.  **Inverted Bottleneck:** Wide $\to$ Narrow $\to$ Wide (like Transformers MLP).
4.  **Fewer Activations:** Replaced ReLU with **GELU**. Used fewer activation layers.
5.  **LayerNorm:** Switched from BatchNorm to LayerNorm.

**Result:** Outperformed Swin Transformer on ImageNet while being simpler and faster.

---

## 💻 Architecture Block

```python
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 1. Depthwise Conv (7x7)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        # 2. Pointwise Expansion (1x1) -> 4x width
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        
        # 3. Pointwise Projection (1x1) -> original width
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        
        # Permute for LayerNorm (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1) 
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # Permute back
        x = x.permute(0, 3, 1, 2)
        
        x = input + x # Residual
        return x
```

---

## 🎓 Interview Focus

1.  **Why did CNNs fall behind ViTs initially?**
    - Mostly due to outdated training recipes (training for 90 epochs vs 300 epochs, no Mixup/CutMix). ConvNeXt proved CNNs are still SOTA.

2.  **BatchNorm vs LayerNorm in Vision?**
    - **BN:** Normalizes across the batch (spatial locations independent). Good for CNNs.
    - **LN:** Normalizes across the channels (per sample). Good for Transformers. ConvNeXt showed LN works for CNNs too.

---

**ConvNeXt: The Empire Strikes Back!**
