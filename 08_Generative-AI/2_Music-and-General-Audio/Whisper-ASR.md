# Whisper ASR

> **Universal Speech Recognition** - OpenAI (2022)

---

## 👂 The Goal

Robust Automatic Speech Recognition (ASR) that works on:
- Multiple languages (99+).
- Noisy audio.
- Accents.
- Technical jargon.

---

## 🏗️ Architecture

Standard **Transformer Encoder-Decoder** (Seq2Seq).

1.  **Input:** Log-Mel Spectrogram (30 seconds).
2.  **Encoder:** Processes audio into a sequence of hidden states.
3.  **Decoder:** Autoregressively predicts text tokens.

---

## 🔑 Weak Supervision at Scale

Whisper wasn't trained on clean, human-labeled datasets (LibriSpeech).
It was trained on **680,000 hours** of audio scraped from the internet.
- **Noisy Labels:** The transcripts were often imperfect (subtitles).
- **Scale:** The sheer volume of data outweighed the noise.

---

## 🛠️ Multitask Format

Whisper performs multiple tasks using special **Task Tokens**:
- `<|startoftranscript|>`
- `<|en|>` (Language ID)
- `<|transcribe|>` or `<|translate|>` (Task)
- `<|notimestamps|>` or `<|0.00|>` (Timestamp prediction)

This allows it to do ASR, Translation, and Voice Activity Detection (VAD) in one model.

---

## 💻 PyTorch Usage (HuggingFace)

```python
from transformers import pipeline

# Load Pipeline
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# Transcribe
result = transcriber("audio.mp3")
print(result["text"])
```

---

## 🎓 Interview Focus

1.  **Why 30-second chunks?**
    - Transformers have a fixed context length. 30 seconds covers most sentences.
    - For longer audio, we use a sliding window or sequential decoding (using the previous 30s text as a prompt for the next 30s).

2.  **Inverse Text Normalization (ITN)?**
    - Whisper outputs "twenty dollars".
    - ITN converts it to "$20".
    - Whisper learns this implicitly from internet subtitles.

---

**Whisper: The standard for open-source ASR!**
