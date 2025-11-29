# Stable Diffusion Architecture

> **Latent Diffusion Models (LDM)** - High-Res Generation on Consumer GPUs

---

## 🐌 The Problem with Pixel Diffusion

Running a U-Net on a $512 \times 512 \times 3$ image for 50 steps is computationally expensive.
Pixel space is huge and redundant.

---

## ⚡ Latent Diffusion (The Solution)

**Idea:** Perform diffusion in a compressed **Latent Space**, not Pixel Space.

1.  **VAE (Variational Autoencoder):**
    - **Encoder:** Compresses Image ($512 \times 512 \times 3$) $\to$ Latent ($64 \times 64 \times 4$). (Factor of 48x smaller).
    - **Decoder:** Decompresses Latent $\to$ Image.

2.  **Diffusion Process:**
    - Train the U-Net to denoise the **Latents** ($64 \times 64$).
    - Much faster!

3.  **Conditioning (CLIP):**
    - How to control generation with text? ("A cat in space").
    - **CLIP Text Encoder:** Converts text to embeddings.
    - **Cross-Attention:** The U-Net attends to the text embeddings via Cross-Attention layers.

---

## 🏗️ The Full Pipeline

1.  **Input:** Text Prompt ("Astronaut riding a horse").
2.  **CLIP:** Encode text to vectors.
3.  **Seed:** Generate random latent noise ($64 \times 64$).
4.  **Denoising Loop (U-Net):**
    - For $t = 50 \to 0$:
    - Predict noise conditioned on Text.
    - Subtract noise.
5.  **VAE Decoder:** Convert final clean latent to Pixel Image ($512 \times 512$).

---

## 💻 HuggingFace Diffusers

```python
from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]

image.save("astronaut.png")
```

---

## 🎓 Interview Focus

1.  **What is Classifier-Free Guidance (CFG)?**
    - A trick to make the image follow the prompt more strictly.
    - We run the model twice: once with the prompt, once with an empty string (unconditioned).
    - $\text{Final} = \text{Uncond} + w \times (\text{Cond} - \text{Uncond})$.
    - $w$ is the Guidance Scale (usually 7.5).

2.  **Why VAE instead of standard Autoencoder?**
    - VAEs enforce a smooth latent space (Gaussian distribution), which is easier for the Diffusion model to learn.

---

**Stable Diffusion: The AI art revolution!**
