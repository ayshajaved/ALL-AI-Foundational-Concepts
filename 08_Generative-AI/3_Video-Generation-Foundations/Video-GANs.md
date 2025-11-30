# Video GANs

> **Early Attempts** - MoCoGAN and VGAN

---

## 📉 The Challenge for GANs

Generating high-res video with GANs is notoriously unstable.
- **Mode Collapse:** The model generates the same motion loop.
- **Blurriness:** Backgrounds become static or blurry.

---

## 🎭 MoCoGAN (Motion and Content GAN)

**Idea:** Decompose Video into **Content** (Static) and **Motion** (Dynamic).

1.  **Content Space ($z_c$):** Sampled once per video. Determines "Who is acting" (Identity, Background).
2.  **Motion Space ($z_m$):** Sampled as a sequence (RNN). Determines "What they are doing".
3.  **Generator:** $G(z_c, z_m^{(t)}) \to \text{Frame}_t$.

**Discriminators:**
- **Image Discriminator:** Is each frame realistic?
- **Video Discriminator:** is the sequence realistic?

---

## 🎞️ TGAN (Temporal GAN)

Uses a **3D Transposed Convolution** Generator.
- Input: Latent vector $z$.
- Output: $T \times H \times W \times C$ volume.
- **Problem:** Extremely memory intensive. Limited to short, low-res clips (16 frames, 64x64).

---

## 🎓 Interview Focus

1.  **Why did Video GANs fail to scale?**
    - 3D Convolutions are expensive.
    - GAN training instability is amplified in 3D.
    - They struggled to separate background (static) from foreground (dynamic) effectively without explicit supervision.

2.  **What replaced them?**
    - **Diffusion Models.** They scale better and handle diverse data distributions without mode collapse.

---

**Video GANs: The pioneers of neural cinema!**
