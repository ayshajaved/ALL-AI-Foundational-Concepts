# WaveNet Deep Dive

> **The Big Bang of Generative Audio** - DeepMind (2016)

---

## 🌊 The Concept

Before WaveNet, TTS sounded robotic (Concatenative synthesis).
WaveNet treats audio generation as a probabilistic task:
**Predict the value of the next sample $x_t$ given all previous samples $x_{<t}$.**

$$ p(\mathbf{x}) = \prod_{t=1}^T p(x_t | x_1, \dots, x_{t-1}) $$

It generates raw audio, sample by sample (16,000 times per second).

---

## 🏗️ Dilated Causal Convolutions

Standard CNNs have a small Receptive Field. To see 1 second of audio (16k samples), you'd need thousands of layers.
**Solution:** Dilated Convolutions.
- Skip inputs with a step size (dilation factor).
- Dilation doubles at each layer: 1, 2, 4, 8, 16...
- **Exponential Receptive Field:** With just a few layers, the network can see thousands of past samples.

**Causal:** The convolution cannot see the future. No padding on the right.

---

## 🔢 Mu-Law Quantization

Predicting a continuous float (16-bit = 65,536 values) is hard.
WaveNet uses **Softmax** classification.
To reduce classes, it compresses audio to 256 values using **$\mu$-law companding** (non-linear quantization).

$$ F(x) = \text{sgn}(x) \frac{\ln(1 + \mu |x|)}{\ln(1 + \mu)} $$

---

## 🚀 The Legacy

WaveNet was slow (autoregressive).
But it proved that **Neural Vocoders** could produce human-level quality.
It paved the way for faster models like **Parallel WaveGAN** and **HiFi-GAN**.

---

## 💻 PyTorch Concept (Dilated Conv)

```python
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.padding = dilation
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=2, 
            dilation=dilation
        )
        
    def forward(self, x):
        # Pad left only (Causal)
        x = nn.functional.pad(x, (self.padding, 0))
        return self.conv(x)

# Stack of dilated convs
layers = []
for i in range(10):
    dilation = 2 ** i
    layers.append(CausalConv1d(32, 32, dilation))
```

---

## 🎓 Interview Focus

1.  **Why is WaveNet slow at inference?**
    - It is autoregressive. To generate 1 second of audio, it must run the network 24,000 times (if 24kHz). It cannot be parallelized.

2.  **What is "Teacher Forcing"?**
    - During training, we feed the *ground truth* past samples to predict the next one. This allows parallel training (all time steps at once).

---

**WaveNet: The model that taught machines to speak!**
