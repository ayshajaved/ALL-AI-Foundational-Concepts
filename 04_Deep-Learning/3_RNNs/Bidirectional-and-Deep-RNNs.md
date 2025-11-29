# Bidirectional and Deep RNNs

> **Enhancing context and capacity** - Looking forward, backward, and deeper

---

## 🎯 Bidirectional RNNs (BiRNN)

Standard RNNs only look at past context. BiRNNs look at **both past and future**.

### Architecture
Two separate RNN layers:
1.  **Forward RNN:** Reads sequence from $x_1$ to $x_T$.
2.  **Backward RNN:** Reads sequence from $x_T$ to $x_1$.

The output at time $t$ is the concatenation of both hidden states:
$$y_t = [h_t^{\rightarrow}; h_t^{\leftarrow}]$$

**Use Cases:**
- Text Classification (Sentiment depends on whole sentence)
- Named Entity Recognition (Context on both sides matters)
- Translation

**Limitation:** Cannot be used for real-time streaming (need future context).

### PyTorch Implementation

```python
import torch.nn as nn

# bidirectional=True
bi_lstm = nn.LSTM(input_size=10, hidden_size=20, 
                  num_layers=1, batch_first=True, 
                  bidirectional=True)

input_seq = torch.randn(5, 8, 10)
output, (hn, cn) = bi_lstm(input_seq)

# Output size doubles (hidden_size * 2)
print(f"BiLSTM Output: {output.shape}") # (5, 8, 40)
```

---

## 📊 Deep (Stacked) RNNs

Stacking multiple RNN layers on top of each other to learn hierarchical features.

### Architecture
- Output of Layer 1 becomes Input of Layer 2.
- $h_t^{(l)} = \text{RNN}(h_t^{(l-1)}, h_{t-1}^{(l)})$

**Capacity:**
- Layer 1: Low-level features (e.g., characters, syntax).
- Layer N: High-level features (e.g., semantics, sentiment).

### PyTorch Implementation

```python
# num_layers=3
deep_lstm = nn.LSTM(input_size=10, hidden_size=20, 
                    num_layers=3, batch_first=True)

output, (hn, cn) = deep_lstm(input_seq)

print(f"Deep LSTM Output: {output.shape}") # (5, 8, 20)
print(f"Hidden States: {hn.shape}")        # (3, 5, 20) -> (layers, batch, hidden)
```

---

## 🔄 Seq2Seq (Encoder-Decoder)

The foundation of Neural Machine Translation.

### Architecture
1.  **Encoder:** RNN that processes input sequence and compresses it into a final "Context Vector" (hidden state).
2.  **Decoder:** RNN that takes the context vector and generates the output sequence step-by-step.

```python
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, source, target):
        # Encode
        _, hidden = self.encoder(source)
        
        # Decode (Teacher forcing loop usually goes here)
        output, _ = self.decoder(target, hidden)
        return output
```

---

## 🎓 Interview Focus

1.  **When to use Bidirectional RNNs?**
    - When the entire sequence is available (offline processing) and future context helps (e.g., NLP). Not for time-series forecasting where future is unknown.

2.  **How does stacking layers help RNNs?**
    - Similar to CNNs/MLPs, depth allows learning more abstract, complex representations of the data.

3.  **What is the bottleneck of Seq2Seq?**
    - The fixed-size context vector must capture ALL information from the source sentence. This limits performance on long sentences (solved by Attention).

---

**Going Deeper and Wider: Maximizing RNN potential!**
