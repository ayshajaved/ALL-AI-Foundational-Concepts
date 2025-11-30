# AudioLM and MusicLM

> **Listening before Speaking** - Google (2022/2023)

---

## 👂 AudioLM: The Foundation

AudioLM proved that you can generate high-quality audio by modeling **Semantic** and **Acoustic** tokens separately.

1.  **Semantic Tokens (w2v-BERT):**
    - Extracted from a self-supervised speech model.
    - Captures *meaning* (phonemes, melody) but discards noise/speaker info.
    - Low sample rate (coarse).

2.  **Acoustic Tokens (SoundStream):**
    - Neural Audio Codec (like VQ-VAE).
    - Captures *fidelity* (timbre, recording quality).
    - High sample rate (fine).

**Generation Process:**
Semantic Tokens $\to$ Coarse Acoustic Tokens $\to$ Fine Acoustic Tokens $\to$ Audio.

---

## 🎸 MusicLM: Text-to-Music

MusicLM builds on AudioLM but adds **Text Conditioning**.

1.  **MuLan (Music-Language Joint Embedding):**
    - A CLIP-like model for Audio.
    - Trains on (Audio, Text) pairs.
    - Maps "Calm violin melody" and an actual violin clip to the same vector space.

2.  **Conditioning:**
    - During training, condition AudioLM on the MuLan audio embedding.
    - During inference, condition AudioLM on the MuLan **text** embedding.
    - Since the embeddings share a space, the model understands the text.

---

## 🎓 Interview Focus

1.  **Why separate Semantic and Acoustic tokens?**
    - If you try to generate Acoustic tokens directly from text, the model gets overwhelmed by details (background noise, mic quality) and loses the melody.
    - Semantic tokens act as a stable bridge.

2.  **What is "Consistency" in MusicLM?**
    - Generating a 5-minute song that stays in the same key and tempo.
    - AudioLM's long context window helps, but it's still a challenge.

---

**MusicLM: The DALL-E of Audio!**
