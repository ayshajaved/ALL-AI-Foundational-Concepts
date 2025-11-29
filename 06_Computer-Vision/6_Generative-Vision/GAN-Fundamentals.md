# GAN Fundamentals (Generative Adversarial Networks)

> **The Minimax Game** - Generator vs Discriminator

---

## 🎭 The Concept (Goodfellow et al., 2014)

Two neural networks competing against each other:

1.  **Generator ($G$):** The Counterfeiter.
    - Input: Random Noise $z$ (Latent Vector).
    - Output: Fake Image $G(z)$.
    - Goal: Fool the Discriminator.

2.  **Discriminator ($D$):** The Police.
    - Input: Image $x$ (Real or Fake).
    - Output: Probability (Real=1, Fake=0).
    - Goal: Distinguish Real from Fake.

---

## 📉 The Loss Function (Minimax)

$$ \min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}} [\log D(x)] + \mathbb{E}_{z \sim p_{z}} [\log (1 - D(G(z)))] $$

- **Discriminator wants to MAXIMIZE:**
    - $\log D(x)$: Predict 1 for real images.
    - $\log (1 - D(G(z)))$: Predict 0 for fake images.
- **Generator wants to MINIMIZE:**
    - $\log (1 - D(G(z)))$: Trick D into predicting 1.

---

## 🏗️ DCGAN (Deep Convolutional GAN)

The architecture that made GANs work for images.
- **Generator:** Uses **Transposed Convolutions** to upsample noise to image.
- **Discriminator:** Uses **Strided Convolutions** to downsample image to probability.
- **Tricks:**
    - Batch Norm in both networks.
    - LeakyReLU in Discriminator.
    - ReLU in Generator (Tanh for output).

---

## 💻 PyTorch Implementation

```python
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.gen = nn.Sequential(
            # Input: N x z_dim x 1 x 1
            nn.ConvTranspose2d(z_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # State: 256 x 4 x 4
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # State: 128 x 8 x 8
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # State: 64 x 16 x 16
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh() # Output: 3 x 32 x 32 (Range -1 to 1)
        )

    def forward(self, x):
        return self.gen(x)
```

---

## 🎓 Interview Focus

1.  **What is Mode Collapse?**
    - The Generator finds *one* image that fools the Discriminator and produces it endlessly (e.g., only generates the number "8").
    - **Fix:** Wasserstein GAN (WGAN), Unrolled GAN.

2.  **Why use Tanh in Generator output?**
    - Images are normalized to $[-1, 1]$. Tanh maps outputs to this range. Sigmoid maps to $[0, 1]$ which causes vanishing gradients in GANs.

3.  **Why are GANs hard to train?**
    - Non-convergence. The two networks oscillate instead of finding a Nash Equilibrium.

---

**GANs: The creative spark of AI!**
