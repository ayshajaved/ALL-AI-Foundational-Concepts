# Talking Head Generation

> **Lip Syncing AI** - Wav2Lip and SadTalker

---

## 🗣️ The Goal

Input:
1.  **Identity Image:** A photo of a face.
2.  **Audio:** A speech clip.

Output:
- A video of the face speaking the audio with perfect lip sync.

---

## 👄 Wav2Lip (2020)

**Core Idea:** A powerful **Sync Discriminator**.
Instead of just checking "Is this real?", it checks "Does this lip shape match this audio?"

1.  **Generator:** Encoder-Decoder. Takes Face + Audio $\to$ Generates Mouth Region.
2.  **Discriminator:** Pre-trained on huge video datasets to detect sync errors.
3.  **Result:** Near-perfect lip sync, but the rest of the face (eyes, head) is static.

---

## 🎭 SadTalker (2023)

Adds **Head Pose** and **Expression** control.
Uses **3DMM (3D Morphable Models)** as an intermediate representation.

1.  **Audio $\to$ Exp/Pose:** Map audio to 3D expression coefficients and head pose angles (using a VAE/Transformer).
2.  **3D $\to$ Video:** Use a generator to render the 3D coefficients back to pixels.
3.  **Result:** Natural head movement and blinking driven by audio.

---

## 💻 PyTorch Concept (Sync Loss)

```python
# SyncNet (Pre-trained)
# Input: [Video_Window, Audio_Window]
# Output: Cosine similarity between Video embedding and Audio embedding

video_embedding = syncnet_visual(mouth_frames)
audio_embedding = syncnet_audio(audio_mfcc)

# We want maximize similarity (minimize distance)
sync_loss = 1 - cosine_similarity(video_embedding, audio_embedding)
```

---

## 🎓 Interview Focus

1.  **What is the "Uncanny Valley"?**
    - If the lip sync is 99% good, the 1% error looks creepy.
    - Wav2Lip crossed the valley for lips, but full-body generation is still hard.

2.  **How to handle high resolution?**
    - Most models generate $96 \times 96$ crops of the mouth.
    - To get 1080p, we paste the generated mouth back onto the original high-res image using super-resolution (GFPGAN).

---

**Talking Heads: The face of AI assistants!**
