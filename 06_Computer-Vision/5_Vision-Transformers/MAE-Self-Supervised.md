# Masked Autoencoders (MAE)

> **BERT for Computer Vision** - Scalable Self-Supervised Learning

---

## 🎯 The Goal

Train a massive Vision Transformer **without labels**.
(Self-Supervised Learning).

---

## 🧩 The Algorithm (He et al., 2021)

**Asymmetric Encoder-Decoder Architecture.**

1.  **Masking:**
    - Split image into patches.
    - Randomly mask a **huge portion (75%)** of patches.
    - Only the *visible* (25%) patches are fed to the Encoder.

2.  **Encoder (ViT-Large):**
    - Processes only visible patches.
    - Very efficient (since 75% of input is gone).

3.  **Decoder (ViT-Small):**
    - Takes encoded visible patches + **Mask Tokens** (placeholders).
    - Reconstructs the original pixels of the masked patches.

4.  **Loss:**
    - MSE (Mean Squared Error) between reconstructed pixels and original pixels.
    - Computed *only* on masked patches.

---

## 🧠 Why 75% Masking?

- **Language (BERT):** Masks 15%. Words are information-dense. "The cat sat on the [MASK]" is easy.
- **Vision (MAE):** Pixels are redundant. If you mask 15%, the model just copies neighbors (interpolation).
- Masking 75% forces the model to understand **high-level semantics** ("I see a tail and an ear, this must be a dog, so I'll fill the middle with fur").

---

## 💻 PyTorch Implementation (Concept)

```python
def forward(self, img):
    patches = self.patchify(img)
    
    # 1. Random Masking
    N = patches.shape[1]
    len_keep = int(N * (1 - 0.75))
    noise = torch.rand(N)
    ids_shuffle = torch.argsort(noise)
    ids_keep = ids_shuffle[:len_keep]
    
    x_masked = patches[:, ids_keep, :]
    
    # 2. Encoder
    latent = self.encoder(x_masked)
    
    # 3. Decoder
    # Add mask tokens back to sequence
    x_full = self.concat_mask_tokens(latent, ids_shuffle)
    pred_pixels = self.decoder(x_full)
    
    return pred_pixels
```

---

## 🎓 Interview Focus

1.  **Why is the Decoder smaller than the Encoder?**
    - The Encoder learns semantic representations (heavy lifting).
    - The Decoder just renders pixels (lighter task).
    - This asymmetry makes training very fast.

2.  **Linear Probing?**
    - After pre-training MAE, we freeze the Encoder and train a linear classifier on top.
    - MAE achieves excellent linear probing accuracy, proving it learned robust features.

---

**MAE: Teaching AI to hallucinate reality!**
