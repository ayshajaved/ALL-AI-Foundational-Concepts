# RNN Fundamentals

> **Modeling sequential data** - From vanilla RNNs to Backpropagation Through Time

---

## 🎯 The Sequence Problem

Standard NNs assume independent inputs. RNNs handle **sequential dependencies**.

**Applications:**
- Time series forecasting (Stock prices, Weather)
- Natural Language Processing (Translation, Sentiment)
- Speech Recognition
- Video Analysis

---

## 📊 Vanilla RNN Architecture

### Recurrent Neuron
A neuron that receives input $x_t$ and its own previous state $h_{t-1}$.

### Equations
$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

Where:
- $h_t$: Hidden state at time $t$ (memory)
- $x_t$: Input at time $t$
- $W_{xh}$: Input-to-hidden weights
- $W_{hh}$: Hidden-to-hidden weights (recurrence)
- $W_{hy}$: Hidden-to-output weights

### Unrolled Computation Graph
```
     y₀       y₁       y₂
     ↑        ↑        ↑
h₀ → h₁   →   h₂   →   h₃ ...
     ↑        ↑        ↑
     x₀       x₁       x₂
```

---

## 💻 Implementation from Scratch

```python
import torch
import torch.nn as nn

class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Weights
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.tanh = nn.Tanh()
        
    def forward(self, input_tensor, hidden_tensor):
        # Concatenate input and hidden state
        combined = torch.cat((input_tensor, hidden_tensor), 1)
        
        # Update hidden state
        hidden = self.i2h(combined)
        hidden = self.tanh(hidden)
        
        # Calculate output
        output = self.i2o(combined)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

# Usage
input_size = 10
hidden_size = 20
output_size = 5
rnn = VanillaRNN(input_size, hidden_size, output_size)

input_tensor = torch.randn(1, 10)
hidden_tensor = rnn.init_hidden()

output, next_hidden = rnn(input_tensor, hidden_tensor)
print(f"Output shape: {output.shape}") # (1, 5)
```

---

## 📉 Backpropagation Through Time (BPTT)

Training RNNs involves unrolling the network across time steps and applying backpropagation.

**The Challenge:**
Gradients flow backward through time.
$$ \frac{\partial L}{\partial W} = \sum_{t} \frac{\partial L_t}{\partial W} $$

### Vanishing Gradient Problem
Since $h_t$ depends on $h_{t-1}$ via multiplication by $W_{hh}$, gradients can vanish or explode exponentially over long sequences.

$$ \frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}} $$

If the eigenvalues of $W_{hh}$ are < 1, gradients vanish (long-term memory loss).
If > 1, gradients explode (instability).

**Solution:**
- **Exploding:** Gradient Clipping
- **Vanishing:** LSTM / GRU architectures (Gating mechanisms)

---

## 🎓 Interview Focus

1.  **Why use RNNs over Feedforward NNs?**
    - RNNs have "memory" (hidden state) to capture temporal dependencies.
    - Can handle variable-length inputs.

2.  **Explain BPTT.**
    - Unrolling the RNN over time steps.
    - Computing gradients for shared weights by summing gradients at each time step.

3.  **Why do gradients vanish in RNNs?**
    - Repeated multiplication of the weight matrix during backpropagation through time steps.
    - Tanh derivative is always < 1.

---

**RNNs: Adding the dimension of time to deep learning!**
