# Emu Video

> **Factorized Generation** - Image-to-Video (Meta)

---

## 🧩 The Factorization Hypothesis

Generating "A cat jumping" directly from text is hard. The model has to decide:
1.  What the cat looks like.
2.  How the cat moves.

**Emu Video** splits this into two explicit steps:
1.  **Text-to-Image:** Generate a high-quality static image of the cat.
2.  **Image-to-Video:** Animate that specific image based on the text prompt.

---

## 🏗️ Architecture

1.  **Model 1 (T2I):** Emu (Latent Diffusion). Generates the starting frame.
2.  **Model 2 (I2V):**
    - Input: Noisy Video Latents + **Conditioning Image** (First frame).
    - The conditioning image is concatenated to the noisy input (channel-wise).
    - The model learns to "unroll" the image into a video.

---

## 📉 Benefits

- **Quality:** The T2I model can focus purely on aesthetics (resolution, lighting).
- **Consistency:** Since the first frame is fixed, the video is grounded. It won't hallucinate a different cat halfway through.
- **Simplicity:** Easier to train two specialized models than one giant monolithic model.

---

## 🎓 Interview Focus

1.  **Why concatenate the image?**
    - It provides a strong signal. The model sees exactly what $t=0$ looks like.
    - It acts as a "visual prompt".

2.  **Relation to SVD (Stable Video Diffusion)?**
    - SVD is also an Image-to-Video model. It takes an image and animates it.
    - This "Image First" workflow is becoming the industry standard for control.

---

**Emu Video: Divide and conquer!**
