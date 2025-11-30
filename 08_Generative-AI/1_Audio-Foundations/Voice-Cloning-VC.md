# Voice Cloning & Conversion

> **Stealing Voices** - Zero-Shot Speaker Adaptation

---

## 🎭 The Goal

**Voice Cloning (TTS):** Generate text in a target person's voice given a short reference audio (3 seconds).
**Voice Conversion (VC):** Change the speaker of an existing audio while keeping the content.

---

## 🧬 Speaker Encodings (d-vectors)

We need to represent "Voice Identity" as a vector.
Train a **Speaker Encoder** network (e.g., GE2E loss) on thousands of speakers.
- Input: Audio clip.
- Output: Fixed-size embedding (e.g., 256-d).
- **Property:** Embeddings of the same speaker are close; different speakers are far.

---

## 🏗️ Multi-Speaker TTS Architecture

Modify a standard TTS (like Tacotron/FastSpeech):
1.  **Input:** Text + **Speaker Embedding**.
2.  **Conditioning:** Concatenate the Speaker Embedding to the Encoder outputs.
3.  **Result:** The decoder generates spectrograms that match the timbre/prosody of the embedding.

**Zero-Shot:**
If the Speaker Encoder is generalized enough, you can feed it audio from a *new* speaker (never seen during training), get an embedding, and the TTS will mimic them.

---

## 🔄 Voice Conversion (VC)

**StarGAN-VC / AutoVC:**
1.  **Encoder:** Compresses audio into content code (removes speaker info).
2.  **Decoder:** Reconstructs audio from content code + **Target Speaker Embedding**.

**Information Bottleneck:**
The encoder must be forced to discard speaker info.
- Reduce bottleneck size.
- Instance Normalization (removes global style).

---

## 🎓 Interview Focus

1.  **What is the difference between Text-to-Speech and Voice Conversion?**
    - TTS: Text $\to$ Audio.
    - VC: Audio $\to$ Audio (Content preserved, Speaker changed).

2.  **Ethical Concerns?**
    - Deepfakes. Voice cloning can be used for fraud.
    - **Watermarking:** Adding imperceptible noise to generated audio to identify it as AI-generated.

3.  **What is RVC (Retrieval-based Voice Conversion)?**
    - A popular VC method. It uses a soft-VC approach with HuBERT features and retrieves the closest feature matches from the target speaker's dataset to improve quality.

---

**Voice Cloning: With great power comes great responsibility!**
