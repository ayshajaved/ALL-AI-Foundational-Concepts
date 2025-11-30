# Audio Signal Processing for AI

> **From Waves to Tensors** - The Language of Sound

---

## 🌊 Sound as a Waveform

Sound is a continuous pressure wave. To process it digitally, we **sample** it.
- **Sample Rate (SR):** How many times per second we measure the wave (e.g., 44.1 kHz, 16 kHz).
- **Bit Depth:** Precision of each sample (e.g., 16-bit).

**Nyquist Theorem:** To capture a frequency $f$, you need a sample rate of at least $2f$.

---

## 📊 The Spectrogram

Raw waveforms (1D) are hard for CNNs to process. We convert them to images (2D).

1.  **Fourier Transform (DFT):** Decomposes a signal into its constituent frequencies.
2.  **STFT (Short-Time Fourier Transform):**
    - Slice audio into small overlapping windows (e.g., 25ms).
    - Compute DFT for each window.
    - Result: **Spectrogram** (Time vs Frequency).

---

## 👂 The Mel Scale

Humans don't hear frequencies linearly. We tell the difference between 100Hz and 200Hz easily, but 10,000Hz and 10,100Hz sound the same.
**Mel Scale:** A perceptual scale of pitches.
$$ m = 2595 \log_{10} (1 + \frac{f}{700}) $$

**Mel-Spectrogram:**
- The standard input for modern Audio AI (Whisper, Stable Audio).
- Maps the linear spectrogram to the Mel scale.

---

## 💻 PyTorch Audio (Torchaudio)

```python
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt

# 1. Load Audio
waveform, sample_rate = torchaudio.load("speech.wav")

# 2. Resample (if needed)
resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
waveform = resampler(waveform)

# 3. Mel Spectrogram Transform
mel_spectrogram = T.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=80
)

mel_spec = mel_spectrogram(waveform)

# 4. Convert to Decibels (Log Scale)
# Audio power is logarithmic (dB)
mel_spec_db = T.AmplitudeToDB()(mel_spec)

# Shape: [Channels, n_mels, Time]
print(mel_spec_db.shape) 
```

---

## 🎓 Interview Focus

1.  **Why use Log-Mel Spectrograms instead of raw audio?**
    - **Compactness:** Compresses high-dimensional audio into a smaller 2D representation.
    - **Perception:** Matches human hearing (Mel scale + Log amplitude).
    - **CNN Compatibility:** Allows using standard image models (ResNet, ViT) on audio.

2.  **What is the "Phase Problem"?**
    - Spectrograms capture **Magnitude** but discard **Phase**.
    - Converting a Spectrogram back to Audio (Griffin-Lim or Vocoder) is non-trivial because the phase info is missing.

---

**Signal Processing: The preprocessing step for all Audio AI!**
