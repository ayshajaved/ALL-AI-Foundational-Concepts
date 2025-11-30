# Modern TTS: FastSpeech 2

> **Need for Speed** - Non-Autoregressive Generation

---

## 🐢 The Problem with Tacotron

Tacotron and WaveNet are **Autoregressive**.
- They generate one token/sample at a time.
- Slow inference.
- **Robustness issues:** Sometimes they repeat words ("Hello... lo... lo") or skip words because the Attention alignment fails.

---

## 🐇 FastSpeech (Transformer TTS)

**Key Idea:** Generate the entire Mel-Spectrogram in **one pass** (Parallel).

**Architecture:**
1.  **Phoneme Encoder:** Transformer Encoder.
2.  **Variance Adaptor:** The magic component.
    - **Duration Predictor:** Predicts how long each phoneme lasts (in frames).
    - **Pitch Predictor:** Predicts the pitch contour.
    - **Energy Predictor:** Predicts volume.
3.  **Mel-Spectrogram Decoder:** Transformer Decoder (Non-autoregressive).

---

## ⏱️ Length Regulator

How do we match the length of text (10 phonemes) to the length of audio (500 frames)?
**Duration Prediction.**
- During training: Use an external aligner (Montreal Forced Aligner) to get ground truth duration for each phoneme.
- During inference: The model predicts duration $d_i$ for phoneme $i$.
- **Expand:** Repeat the hidden state of phoneme $i$, $d_i$ times.

---

## 💻 Comparison

| Feature | Tacotron 2 | FastSpeech 2 |
| :--- | :--- | :--- |
| **Type** | Autoregressive (RNN) | Non-Autoregressive (Transformer) |
| **Speed** | Slow (Real-time x1) | Fast (Real-time x50) |
| **Robustness** | Prone to skipping/repeating | Very Stable (Hard alignment) |
| **Control** | Hard to control speed/pitch | Explicit control (edit duration/pitch) |

---

## 🎓 Interview Focus

1.  **How does FastSpeech handle the "One-to-Many" problem?**
    - One text can be spoken in many ways (fast/slow, happy/sad).
    - Tacotron learns an average (blurry).
    - FastSpeech explicitly models the variance (Pitch, Energy, Duration) as inputs, making the mapping deterministic given these inputs.

2.  **Why do we need an external aligner?**
    - Since FastSpeech is non-autoregressive, it doesn't learn alignment implicitly via Attention. It needs supervision on "which phoneme corresponds to which audio frames."

---

**FastSpeech: The engine behind modern real-time assistants!**
