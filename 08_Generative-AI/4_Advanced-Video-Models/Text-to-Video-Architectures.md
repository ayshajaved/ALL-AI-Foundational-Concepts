# Text-to-Video Architectures

> **"A teddy bear swimming in 4k"** - Make-A-Video, Imagen Video, Sora

---

## 🎥 Make-A-Video (Meta)

**Philosophy:** Don't retrain from scratch. Adapt a pre-trained Text-to-Image (T2I) model.
1.  **Base:** Pre-trained T2I model (DALLE-2 equivalent).
2.  **Adaptation:** Add Temporal Attention layers.
3.  **Super-Resolution:**
    - Spatial SR: $64 \times 64 \to 256 \times 256$.
    - Temporal SR (Frame Interpolation): $16$ fps $\to$ $60$ fps.

---

## 🖼️ Imagen Video (Google)

**Philosophy:** Cascaded Diffusion Models.
A pipeline of 7 models!
1.  **Base:** Generates $16$ frames at $24 \times 48$ resolution.
2.  **SSR (Spatial Super-Res):** Upscales to $1280 \times 768$.
3.  **TSR (Temporal Super-Res):** Increases frame count to 128 frames.

**Key Feature:** **v-prediction** parameterization (instead of $\epsilon$-prediction) for better color stability and convergence.

---

## 🌌 Sora (OpenAI)

**Philosophy:** Video is just a sequence of patches.
1.  **VAE:** Compresses video into a latent spacetime volume.
2.  **Patchify:** Cuts the volume into 3D patches ("Spacetime tokens").
3.  **Transformer (DiT):** A massive Diffusion Transformer processes these tokens.
    - Handles variable resolutions and aspect ratios natively (no cropping).
    - Scales effectively with compute (Scaling Laws).

---

## 🎓 Interview Focus

1.  **Why Cascaded Models?**
    - Generating high-res video directly is too memory intensive.
    - Splitting the task (Base generation $\to$ Upscaling $\to$ Interpolation) makes it manageable.

2.  **DiT vs U-Net for Video?**
    - **U-Net:** Hard to handle variable aspect ratios. Good inductive bias for images.
    - **DiT (Transformer):** Flexible (patches). Scales better. Sora proved DiT is the future for video.

---

**Text-to-Video: The new frontier of storytelling!**
