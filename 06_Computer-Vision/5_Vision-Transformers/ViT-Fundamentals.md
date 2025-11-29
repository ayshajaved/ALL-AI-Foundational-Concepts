# Vision Transformer (ViT) Fundamentals

> **Images are worth $16 \times 16$ words** - The Paradigm Shift

---

## 🦖 The CNN Monopoly

For 10 years (2012-2020), Computer Vision = CNN (ResNet, VGG).
**Inductive Bias of CNNs:**
1.  **Locality:** Pixels are related to their neighbors.
2.  **Translation Invariance:** A cat in the top-left is the same as a cat in the bottom-right.

**Transformers:** Have *no* such biases. They learn relationships globally.

---

## 🖼️ ViT Architecture (Dosovitskiy et al., 2021)

How to feed a 2D image into a 1D Transformer?

### 1. Patchify
Split image $(H, W, C)$ into fixed-size patches $(P, P)$.
- Image: $224 \times 224$. Patch: $16 \times 16$.
- Number of Patches $N = (224/16)^2 = 196$.
- Flatten each patch into a vector of size $16 \times 16 \times 3 = 768$.

### 2. Linear Projection
Map each flattened patch to a latent vector $D$ (Embedding Dimension).
Result: Sequence of $N$ vectors.

### 3. Positional Embeddings
Since Transformers are permutation invariant (bag of words), we add learnable vectors to indicate position.
"This patch is top-left", "This patch is center".

### 4. The Class Token (`[CLS]`)
Prepend a learnable vector (`[CLS]`) to the sequence.
The output state of this token serves as the image representation.

### 5. Transformer Encoder
Standard Multi-Head Self-Attention (MSA) + MLP layers.

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

Every patch attends to *every other patch* (Global Receptive Field) from Layer 1.

---

## 💻 PyTorch Implementation

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 3, 224, 224) -> (B, 768, 14, 14)
        x = self.proj(x)
        # Flatten: (B, 768, 196) -> Transpose: (B, 196, 768)
        x = x.flatten(2).transpose(1, 2)
        return x

class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbedding()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.pos_embed = nn.Parameter(torch.zeros(1, 196 + 1, 768))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=12), num_layers=12
        )
        self.head = nn.Linear(768, 1000)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        # Append CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add Positional Embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.encoder(x)
        
        # Classify using only CLS token output
        return self.head(x[:, 0])
```

---

## 🎓 Interview Focus

1.  **ViT vs ResNet: Data Efficiency?**
    - **ResNet:** Performs better on small datasets (ImageNet-1k) due to inductive bias.
    - **ViT:** Performs better on massive datasets (JFT-300M). It needs more data to *learn* the inductive biases (like locality) that CNNs have hard-coded.

2.  **What is the computational complexity of ViT?**
    - $O(N^2)$ with respect to number of patches.
    - If you double image resolution, patches quadruple $\to$ compute increases 16x.

---

**ViT: Breaking the grid!**
