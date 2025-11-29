# Swin Transformer

> **Hierarchical Vision Transformer** - Bringing back the Pyramid

---

## 🧱 The Problem with ViT

1.  **Quadratic Complexity:** $O(N^2)$. Cannot handle high-resolution images (e.g., $1024 \times 1024$ for segmentation).
2.  **Single Scale:** ViT processes patches at a single resolution ($16 \times 16$). CNNs use pyramids (Feature Maps get smaller and deeper).

---

## 🏗️ Swin Architecture (Shifted Windows)

**Key Idea:** Compute Self-Attention only within small local windows (e.g., $7 \times 7$ patches).
Complexity becomes **Linear** $O(N)$ with respect to image size.

### 1. Window Attention (W-MSA)
Divide image into non-overlapping windows.
Perform attention *inside* each window independently.
**Problem:** No communication between windows.

### 2. Shifted Window Attention (SW-MSA)
In the next layer, shift the window partitioning by half the window size.
This bridges the windows, allowing information to flow across boundaries.

### 3. Hierarchical Structure
Like a ResNet:
- Stage 1: $H/4 \times W/4$
- Stage 2: $H/8 \times W/8$ (Patch Merging)
- Stage 3: $H/16 \times W/16$
- Stage 4: $H/32 \times W/32$

This makes Swin compatible with **FPN** (Feature Pyramid Networks) for Object Detection and Segmentation.

---

## 💻 Patch Merging (Downsampling)

Equivalent to Pooling in CNNs.
Takes a $2 \times 2$ group of patches and concatenates them.
- Channels: $C \to 4C$.
- Resolution: $H \times W \to H/2 \times W/2$.
- Linear layer reduces channels $4C \to 2C$.

---

## 🎓 Interview Focus

1.  **Why is Swin better than ViT for Detection?**
    - Detection requires multi-scale features (small objects need high res, large objects need global context). Swin provides this hierarchy. ViT is columnar (single scale).

2.  **Explain Shifted Windows.**
    - Layer $l$: Regular grid.
    - Layer $l+1$: Grid shifted by $(M/2, M/2)$.
    - This creates cross-window connections efficiently without global attention.

3.  **Relative Positional Encoding?**
    - Swin adds a bias to the attention score based on the relative position of patches within a window, rather than absolute position.

---

**Swin: The best of both worlds (CNN + Transformer)!**
