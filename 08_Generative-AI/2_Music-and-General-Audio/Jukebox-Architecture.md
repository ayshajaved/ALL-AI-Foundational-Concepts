# Jukebox Architecture

> **The GPT of Music** - OpenAI (2020)

---

## 🎵 The Challenge

Generating raw audio with long-term structure (minutes, not seconds).
WaveNet (Autoregressive on pixels) loses coherence after a few seconds.

---

## 🏗️ VQ-VAE (Vector Quantized Variational Autoencoder)

Jukebox compresses audio into discrete codes using a **Hierarchical VQ-VAE**.

1.  **Compression:**
    - **Level 1 (Top):** Compresses 128x. Captures high-level structure (Genre, Melody).
    - **Level 2 (Middle):** Compresses 32x.
    - **Level 3 (Bottom):** Compresses 8x. Captures fine acoustic details.

2.  **Quantization:**
    - Maps continuous vectors to a Codebook (Vocabulary of 2048 vectors).
    - Result: Audio becomes a sequence of integers (Tokens).

---

## 🧠 The Prior (Sparse Transformers)

Once audio is tokenized, we train **Transformers** to predict the next token (like GPT).

1.  **Top-Level Prior:** Generates the coarse structure (Level 1 tokens) conditioned on Artist/Genre/Lyrics.
2.  **Upsamplers:** Generate Level 2 tokens conditioned on Level 1. Then Level 3 conditioned on Level 2.
3.  **Decoder:** VQ-VAE Decoder converts Level 3 tokens back to raw audio.

---

## 🎤 Lyrics Conditioning

Jukebox aligns lyrics to audio using a specialized attention mechanism.
It allows "singing" specific words at specific times.

---

## 🎓 Interview Focus

1.  **Why Hierarchical?**
    - Generating sample-by-sample is too hard.
    - Generating "Bar-by-Bar" (Level 1) is easier. Then fill in the details (Level 2/3).
    - It's like sketching a painting before coloring it.

2.  **What is the downside of Jukebox?**
    - **Slow:** It takes hours to generate 1 minute of music.
    - **Noisy:** The VQ-VAE reconstruction is not perfect (artifacts).

---

**Jukebox: The first model to sing intelligible lyrics!**
