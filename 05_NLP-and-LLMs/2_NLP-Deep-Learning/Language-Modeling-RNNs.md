# Language Modeling with RNNs

> **Predicting the next word** - The foundation of Generative AI

---

## 🎯 The Task

Given a sequence $x_1, x_2, ..., x_t$, predict $x_{t+1}$.
$$ P(w_1, w_2, ..., w_n) = \prod_{i=1}^n P(w_i | w_{<i}) $$

**Types:**
1.  **Character-Level:** Predict next char (`h` $\to$ `e` $\to$ `l` $\to$ `l` $\to$ `o`). Small vocab, long sequences.
2.  **Word-Level:** Predict next word. Large vocab, shorter sequences.

---

## 🏗️ Architecture

Input: `[The, cat, sat]`
Target: `[cat, sat, on]`

1.  Embedding
2.  LSTM (Unrolled)
3.  Linear $\to$ Vocab Size
4.  CrossEntropyLoss

---

## 💻 Text Generation Loop

How to generate text once the model is trained.

```python
import torch.nn.functional as F

def generate(model, start_str, len_generated=100, temperature=1.0):
    model.eval()
    input_ids = tokenizer.encode(start_str)
    hidden = None
    
    generated_ids = []
    
    for _ in range(len_generated):
        # Prepare input (batch_size=1)
        x = torch.tensor([input_ids]).to(device)
        
        # Forward
        # output: [1, seq_len, vocab_size]
        output, hidden = model(x, hidden)
        
        # Get logits for last token
        logits = output[0, -1, :] / temperature
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).item()
        
        generated_ids.append(next_token_id)
        input_ids.append(next_token_id) # Autoregressive: append output to input
        
    return tokenizer.decode(generated_ids)
```

---

## 🌡️ Temperature Sampling

Controls randomness.
- **Low Temp ($<1.0$):** Conservative. Picks most likely words. Repetitive.
- **High Temp ($>1.0$):** Creative. Flattens distribution. Can be incoherent.

$$ P_i = \frac{\exp(z_i / T)}{\sum \exp(z_j / T)} $$

---

## 🎓 Interview Focus

1.  **What is "Teacher Forcing"?**
    - During training, we feed the *ground truth* token as input for the next step, not the model's own prediction.
    - Speeds up convergence.

2.  **Greedy Decoding vs Sampling?**
    - **Greedy:** Always pick max probability. Boring, repetitive.
    - **Sampling:** Pick based on probability distribution. Diverse.

3.  **Why RNNs for LM?**
    - They can theoretically handle infinite context (via hidden state).
    - *Reality:* They forget after ~100 tokens. Transformers replaced them.

---

**Language Modeling: The pre-training objective of the gods!**
