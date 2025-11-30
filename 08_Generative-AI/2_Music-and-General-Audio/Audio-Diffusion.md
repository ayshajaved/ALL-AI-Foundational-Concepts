# Audio Diffusion

> **Painting with Sound** - Riffusion and Stable Audio

---

## 🎨 Spectrogram Diffusion (Riffusion)

**Idea:** A Spectrogram is just an image. Can we fine-tune Stable Diffusion on spectrograms?

1.  **Data:** Convert Audio $\to$ Spectrogram Image.
2.  **Training:** Fine-tune Stable Diffusion (UNet) to generate these "images" from text prompts.
3.  **Inference:** Generate image $\to$ Griffin-Lim/Vocoder $\to$ Audio.

**Result:** Surprisingly good music generation, capable of looping and interpolation.

---

## 🌌 Latent Diffusion (Stable Audio)

Generating pixel-perfect spectrograms is hard (phase issues).
**Stable Audio** uses Latent Diffusion (like LDMs for images).

1.  **VAE (Variational Autoencoder):** Compresses audio into a continuous latent space.
    - Unlike Jukebox (Discrete VQ-VAE), this is continuous.
2.  **Diffusion Model (DiT):** A Diffusion Transformer operates in this latent space.
3.  **Conditioning:** CLAP (Contrastive Language-Audio Pretraining) embeddings for text guidance.

---

## ⏱️ Timing Conditioning

Stable Audio adds a **"Seconds Start"** and **"Seconds Total"** token.
- Allows generating a "30-second intro" or a "3-minute song".
- Fixes the issue of fixed-length generation.

---

## 🎓 Interview Focus

1.  **Image Diffusion vs Audio Diffusion?**
    - **Image:** Spatial correlation (pixels near each other matter).
    - **Audio:** Temporal correlation (patterns repeat over time).
    - Using rectangular kernels (e.g., $1 \times 9$) in CNNs helps capture temporal structure better than square kernels ($3 \times 3$).

2.  **Why is Phase reconstruction hard for Riffusion?**
    - Riffusion generates magnitude spectrograms. It guesses the phase. This leads to a "metallic" or "phaser" artifact in the sound.
    - Latent Diffusion (Stable Audio) avoids this by decoding from a VAE trained to reconstruct high-fidelity audio.

---

**Audio Diffusion: The current SOTA for sound generation!**
