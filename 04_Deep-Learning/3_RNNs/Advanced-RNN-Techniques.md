# Advanced RNN Techniques

> **Optimizing performance and stability** - Attention, Teacher Forcing, and more

---

## 🎯 Attention Mechanism (Intro)

*Note: Detailed coverage in Transformers section. Here we focus on RNN context.*

**Problem:** Seq2Seq context vector bottleneck.
**Solution:** Allow decoder to "look at" all encoder hidden states, not just the last one.

$$c_t = \sum_{j=1}^{T} \alpha_{tj} h_j$$
Where $\alpha_{tj}$ is the attention weight (importance of encoder state $j$ for decoder step $t$).

---

## 🏫 Teacher Forcing

A training strategy for autoregressive models (RNNs generating sequences).

**Standard Training:** Feed model's *own prediction* at $t$ as input for $t+1$.
**Problem:** Errors accumulate (exposure bias). Slow convergence.

**Teacher Forcing:** Feed the **ground truth** (actual target) at $t$ as input for $t+1$.

```python
# Pseudo-code
use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

if use_teacher_forcing:
    # Feed the target as the next input
    decoder_input = target_tensor[t] 
else:
    # Feed the model's own prediction
    decoder_input = model_output
```

**Pros:** Faster convergence.
**Cons:** Train/Test mismatch (model never learns to recover from its own mistakes).

---

## ✂️ Gradient Clipping

**Problem:** Exploding gradients (gradients > 1 accumulate exponentially).
**Solution:** Cap the norm of the gradient vector.

```python
# PyTorch
optimizer.zero_grad()
loss.backward()

# Clip gradients in-place to max_norm=1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

optimizer.step()
```

---

## 📉 Truncated BPTT

**Problem:** Backpropagating through very long sequences (e.g., 10k steps) is memory intensive and slow.
**Solution:** Process sequence in chunks (e.g., 50 steps). Detach hidden state between chunks.

```python
# Truncated BPTT Loop
hidden = None
for i in range(0, len(long_seq), chunk_size):
    x_chunk = long_seq[i : i+chunk_size]
    y_chunk = targets[i : i+chunk_size]
    
    # Detach hidden state from previous computation graph
    if hidden is not None:
        hidden = hidden.detach()
        
    output, hidden = model(x_chunk, hidden)
    loss = criterion(output, y_chunk)
    loss.backward()
    optimizer.step()
```

---

## 🎲 Scheduled Sampling

A middle ground between Teacher Forcing and Free Running.
- Start training with 100% Teacher Forcing.
- Gradually decrease forcing ratio and increase model's reliance on its own predictions.
- Helps bridge the Train/Test gap.

---

## 🎓 Interview Focus

1.  **Why use Gradient Clipping?**
    - To prevent exploding gradients from destabilizing training (NaN weights).

2.  **What is Exposure Bias?**
    - The discrepancy where a model is trained on ground truth inputs (Teacher Forcing) but must generate based on its own potentially erroneous predictions at inference time.

3.  **Why detach hidden states in Truncated BPTT?**
    - To stop gradients from flowing back beyond the current chunk, saving memory and preventing vanishing gradients over excessive lengths.

---

**Advanced techniques for robust RNN training!**
