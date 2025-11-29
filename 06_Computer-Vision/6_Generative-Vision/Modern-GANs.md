# Modern GANs

> **Photorealism & Control** - StyleGAN and CycleGAN

---

## 💇‍♀️ StyleGAN (NVIDIA)

**Goal:** Generate high-resolution, photorealistic faces with control over features (Pose, Hair, Eyes).

**Key Innovation: Style Mixing**
Instead of feeding $z$ directly into the input layer:
1.  **Mapping Network:** Converts $z \to w$ (Intermediate Latent Space).
2.  **Synthesis Network:** Starts from a constant learned input.
3.  **AdaIN (Adaptive Instance Normalization):** Injects $w$ (style) at every layer.
    - Early layers: Control coarse features (Pose, Face Shape).
    - Middle layers: Control facial features (Eyes, Mouth).
    - Fine layers: Control textures (Skin pores, Hair color).

**Result:** You can mix the "Pose" of Person A with the "Hair" of Person B.

---

## 🔄 CycleGAN (Unpaired Translation)

**Goal:** Translate Domain $X \to Y$ (e.g., Horse $\to$ Zebra) **without paired data**.
We don't have photos of the *exact same* horse as a zebra.

**Cycle Consistency Loss:**
If we translate Horse $\to$ Zebra $\to$ Horse, we should get the original horse back.

$$ L_{cyc} = || G_{Y \to X}(G_{X \to Y}(x)) - x ||_1 $$

**Architecture:**
- Two Generators: $G_{X \to Y}$ and $G_{Y \to X}$.
- Two Discriminators: $D_X$ and $D_Y$.

---

## 💻 Pix2Pix (Paired Translation)

Requires paired data (e.g., Sketch $\to$ Photo).
**Loss:** GAN Loss + L1 Loss (Pixel-wise difference).
**PatchGAN Discriminator:** Classifies $N \times N$ patches as Real/Fake instead of the whole image. Captures high-frequency texture details.

---

## 🎓 Interview Focus

1.  **Why does StyleGAN use a Mapping Network ($z \to w$)?**
    - The input space $z$ (Gaussian) is entangled. The mapping network "unwraps" it into $w$, making features linear and separable (Disentanglement).

2.  **Difference between Pix2Pix and CycleGAN?**
    - **Pix2Pix:** Supervised (Paired data). Sketch $\to$ Photo.
    - **CycleGAN:** Unsupervised (Unpaired data). Summer $\to$ Winter.

3.  **What is AdaIN?**
    - Normalizes feature maps to mean 0, std 1, then scales and shifts them using parameters learned from the style vector $w$.
    - $AdaIN(x, y) = \sigma(y) \left( \frac{x - \mu(x)}{\sigma(x)} \right) + \mu(y)$

---

**Modern GANs: Dreaming with eyes open!**
