# Prioritized Experience Replay (PER)

> **Not all memories are created equal** - Learning from Surprise

---

## 🧠 The Intuition

Standard DQN samples from the Replay Buffer **uniformly**.
But we learn most from transitions where we were **wrong** (high TD error).
"I thought this move was safe, but I died." $\to$ High Error $\to$ Important!

**Idea:** Sample transitions with probability proportional to their TD error.

---

## ⚙️ The Algorithm

1.  **Priority ($p_i$):**
    $$ p_i = |\delta_i| + \epsilon $$
    - $\delta_i$: TD Error.
    - $\epsilon$: Small constant (to ensure zero-error transitions are still sampled occasionally).

2.  **Probability ($P(i)$):**
    $$ P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha} $$
    - $\alpha$: How much prioritization to use (0 = Uniform, 1 = Full Priority).

3.  **Importance Sampling Weights ($w_i$):**
    Prioritization introduces **bias** (we see "hard" samples more often).
    To correct this, we down-weight the loss for high-priority samples.
    $$ w_i = \left( \frac{1}{N \cdot P(i)} \right)^\beta $$

---

## 🏗️ Implementation Details (SumTree)

Calculating $P(i)$ requires summing all priorities. $O(N)$. Too slow.
**SumTree Data Structure:**
- Binary tree where parent = sum of children.
- Update priority: $O(\log N)$.
- Sample: $O(\log N)$.

---

## 🎓 Interview Focus

1.  **Why do we need Importance Sampling weights?**
    - Because by changing the sampling distribution, we are no longer estimating the expected value under the true distribution. The weights correct the gradient magnitude so the optimization step size is appropriate.

2.  **What happens if $\beta=1$?**
    - Full compensation. The gradient updates behave as if we sampled uniformly, but we focused computation on the most informative samples. Usually, we anneal $\beta$ from $0.4 \to 1.0$.

---

**PER: Focusing on what matters!**
