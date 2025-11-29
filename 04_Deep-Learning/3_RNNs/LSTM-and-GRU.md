# LSTM and GRU

> **Solving the vanishing gradient problem** - Long Short-Term Memory and Gated Recurrent Units

---

## 🎯 Long Short-Term Memory (LSTM)

Designed to preserve gradients over long sequences using a **cell state** ($C_t$) and **gates**.

### Architecture
Three gates control information flow:
1.  **Forget Gate ($f_t$):** What to throw away from cell state.
2.  **Input Gate ($i_t$):** What new information to store.
3.  **Output Gate ($o_t$):** What to output (hidden state).

### Equations
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$
$$C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$$
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t * \tanh(C_t)$$

**Key Insight:** The cell state $C_t$ has a linear update path ($f_t * C_{t-1} + ...$), allowing gradients to flow without vanishing.

---

## 📊 Gated Recurrent Unit (GRU)

A simplified version of LSTM. Merges cell state and hidden state.

### Architecture
Two gates:
1.  **Reset Gate ($r_t$):** How much of the past to ignore.
2.  **Update Gate ($z_t$):** Balance between past and new information.

### Equations
$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$$
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$$
$$\tilde{h}_t = \tanh(W \cdot [r_t * h_{t-1}, x_t])$$
$$h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t$$

**Pros:** Fewer parameters, faster training, often comparable performance.

---

## 💻 PyTorch Implementation

```python
import torch
import torch.nn as nn

# LSTM
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=1, batch_first=True)

# GRU
gru = nn.GRU(input_size=10, hidden_size=20, num_layers=1, batch_first=True)

# Input: (batch, seq_len, input_size)
input_seq = torch.randn(5, 8, 10) 

# Forward LSTM
# h0, c0 default to zeros if not provided
output, (hn, cn) = lstm(input_seq)
print(f"LSTM Output: {output.shape}") # (5, 8, 20)
print(f"LSTM Hidden: {hn.shape}")     # (1, 5, 20)

# Forward GRU
output, hn = gru(input_seq)
print(f"GRU Output: {output.shape}")  # (5, 8, 20)
```

---

## ⚖️ LSTM vs GRU vs Vanilla RNN

| Feature | Vanilla RNN | LSTM | GRU |
| :--- | :--- | :--- | :--- |
| **Parameters** | Low | High (4x RNN) | Medium (3x RNN) |
| **Training Speed** | Fast | Slow | Medium |
| **Long Dependencies** | Poor | Excellent | Good |
| **Complexity** | Simple | Complex | Moderate |
| **Use Case** | Short sequences | Complex, long sequences | General purpose |

---

## 🎓 Interview Focus

1.  **How does LSTM solve vanishing gradients?**
    - The additive update of the cell state ($C_t = f_t C_{t-1} + ...$) creates a "gradient superhighway" where gradients can flow unchanged if $f_t \approx 1$.

2.  **Difference between LSTM and GRU?**
    - LSTM has 3 gates and separate cell/hidden states.
    - GRU has 2 gates and merged state. GRU is computationally cheaper.

3.  **What is the role of the forget gate?**
    - It allows the network to reset its memory (e.g., at the end of a sentence) to prevent irrelevant information from persisting.

---

## 🧠 Toy Task: The "Copy" Problem

Prove LSTM is better than RNN.
**Task:** Remember a sequence of random integers and repeat it after a delay.

```python
# 1. Generate Data
# Input:  [1, 3, 5, 0, 0, 0] (0 is padding/delay)
# Target: [0, 0, 0, 1, 3, 5]

# 2. Train both models
rnn = VanillaRNN(...)
lstm = LSTM(...)

# 3. Result
# RNN Loss: Stagnates (Gradients vanish over the delay)
# LSTM Loss: Converges to zero (Cell state carries info)
```

**Why it happens:**
In Vanilla RNN, the gradient $\frac{\partial h_t}{\partial h_{t-k}}$ decays exponentially with $k$.
In LSTM, the gradient flows through the cell state $C_t$ with a factor of $\approx 1$ (if forget gate is open), preserving the signal indefinitely.

---

**LSTM & GRU: The workhorses of sequence modeling!**
