# Generalized Advantage Estimation (GAE)

> **Tuning the Bias-Variance Knob** - The $\lambda$ Parameter

---

## 🎛️ The Problem

We need to estimate the Advantage $A(s, a)$.
We have two estimators:
1.  **TD(0) Error:** $A \approx R_t + \gamma V(S_{t+1}) - V(S_t)$. (High Bias, Low Variance).
2.  **Monte Carlo Error:** $A \approx G_t - V(S_t)$. (Low Bias, High Variance).

Can we find a middle ground?

---

## 📐 GAE($\lambda$)

Exponentially weighted average of $k$-step TD errors.

Define TD residual $\delta_t = R_t + \gamma V(S_{t+1}) - V(S_t)$.

$$ A_t^{GAE(\lambda)} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l} $$

- **$\lambda = 0$:** $A_t = \delta_t$. (TD(0) - High Bias).
- **$\lambda = 1$:** $A_t = \sum \gamma^l \delta_{t+l} = G_t - V(S_t)$. (Monte Carlo - High Variance).

**Typical Value:** $\lambda = 0.95$.
Best of both worlds.

---

## 💻 Calculation (Recursive)

We can compute GAE efficiently backwards from the end of the episode.

$$ A_t^{GAE} = \delta_t + (\gamma \lambda) A_{t+1}^{GAE} $$

```python
def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    gae = 0
    returns = []
    
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_values[step] * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        returns.insert(0, gae + values[step])
        
    return returns
```

---

## 🎓 Interview Focus

1.  **What does $\lambda$ control?**
    - It controls the trade-off between bias and variance in the advantage estimate.
    - Lower $\lambda$ relies more on the Critic (Value Function).
    - Higher $\lambda$ relies more on the actual Rewards.

2.  **Where is GAE used?**
    - Almost everywhere in modern Policy Gradients (PPO, TRPO, A2C). It is a standard component.

---

**GAE: The secret sauce of stable training!**
