# HuggingFace Audio Course

> **The Transformer Library for Sound** - Pipelines and Datasets

---

## 🎧 The Ecosystem

HuggingFace isn't just for NLP. It has a massive ecosystem for Audio.
- **`datasets`:** Load audio datasets (LibriSpeech, Common Voice) with one line.
- **`transformers`:** Pre-trained models (Wav2Vec2, Whisper, HuBERT).
- **`evaluate`:** Metrics (WER, CER).

---

## 🛠️ The `pipeline()` Abstraction

The easiest way to use Audio models.

```python
from transformers import pipeline
from datasets import load_dataset

# 1. Automatic Speech Recognition (ASR)
asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
result = asr("audio.mp3")
print(result["text"])

# 2. Audio Classification (Emotion Recognition)
classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-ks")
label = classifier("command.wav")
print(label)

# 3. Text-to-Speech (TTS)
tts = pipeline("text-to-speech", model="suno/bark-small")
audio = tts("Hello world!")
# Returns a numpy array of audio
```

---

## 📊 Handling Audio Data

Audio comes in different sample rates. Models expect a specific rate (usually 16kHz).
**Always resample!**

```python
from datasets import Audio

# Load dataset
dataset = load_dataset("polyai_minds14", "en-US", split="train")

# Cast column to Audio feature (Automatic Resampling)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Accessing an item automatically decodes and resamples it
sample = dataset[0]["audio"]
print(sample["array"].shape, sample["sampling_rate"])
```

---

## 🎓 Interview Focus

1.  **What is CTC (Connectionist Temporal Classification)?**
    - A loss function used for ASR (Wav2Vec2).
    - Allows the network to output a sequence of characters shorter than the audio frames, handling alignment automatically (by outputting "blank" tokens).

2.  **Wav2Vec2 vs Whisper?**
    - **Wav2Vec2:** Encoder-only. Trained with CTC or Finetuning. Good for classification/ASR.
    - **Whisper:** Encoder-Decoder. Seq2Seq. Good for ASR/Translation.

---

**HuggingFace Audio: NLP tools for the Audio world!**
