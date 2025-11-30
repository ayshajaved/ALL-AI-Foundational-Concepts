# Text-to-Speech (TTS) Fundamentals

> **Giving Machines a Voice** - The TTS Pipeline

---

## 🗣️ The Standard Pipeline

Modern TTS is usually a two-stage process:

1.  **Acoustic Model (Text $\to$ Spectrogram):**
    - Input: "Hello World" (Phonemes).
    - Output: Mel-Spectrogram (Acoustic features).
    - *Examples:* Tacotron 2, FastSpeech 2.

2.  **Vocoder (Spectrogram $\to$ Audio):**
    - Input: Mel-Spectrogram.
    - Output: Raw Waveform.
    - *Examples:* WaveNet, HiFi-GAN, MelGAN.

---

## 🏗️ Tacotron 2 (The Classic)

**Architecture:** Seq2Seq with Attention (Encoder-Decoder).

1.  **Encoder:**
    - Character Embeddings $\to$ 3 Convolutional Layers $\to$ Bi-LSTM.
    - Encodes text into a context vector.

2.  **Attention (Location-Sensitive):**
    - Decides which part of the text to focus on while generating the next spectrogram frame.
    - Crucial for alignment (Speech speed varies).

3.  **Decoder:**
    - LSTM-based. Generates one spectrogram frame at a time (Autoregressive).

---

## 🔡 Text Processing (Grapheme vs Phoneme)

- **Graphemes:** Raw letters ("C", "a", "t").
    - *Problem:* English is weird. "Read" (present) vs "Read" (past). "Through" vs "Tough".
- **Phonemes:** Sounds (/k/, /ae/, /t/).
    - *Solution:* Use a **G2P (Grapheme-to-Phoneme)** converter (e.g., CMU Dict) before feeding to the model.
    - "Hello" $\to$ `HH AH0 L OW1`.

---

## 💻 G2P Implementation

```python
from g2p_en import G2p

g2p = G2p()
text = "Artificial Intelligence is fascinating."
phonemes = g2p(text)

print(phonemes)
# Output: ['AA1', 'R', 'T', 'AH0', 'F', 'IH1', 'SH', 'AH0', 'L', ' ', 'IH2', 'N', 'T', 'EH1', 'L', 'AH0', 'JH', 'AH0', 'N', 'S', ' ', 'IH1', 'Z', ' ', 'F', 'AE1', 'S', 'AH0', 'N', 'EY2', 'T', 'IH0', 'NG', '.']
```

---

## 🎓 Interview Focus

1.  **Why do we need a Vocoder?**
    - The Acoustic Model predicts the *magnitude* of frequencies (Spectrogram). It doesn't predict the fine-grained *phase* needed to generate a clean wave. The Vocoder solves this phase reconstruction problem.

2.  **End-to-End TTS (VITS)?**
    - Newer models like **VITS** (Conditional Variational Autoencoder with Adversarial Learning) combine the Acoustic Model and Vocoder into one network, training directly from Text to Waveform.

---

**TTS: The art of synthetic speech!**
