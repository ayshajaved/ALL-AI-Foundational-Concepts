# Monte Carlo Methods

> **Learning from Experience** - No Model Required

---

## 🎲 The Idea

We don't know $P(s'|s,a)$. We can't do the sum $\sum_{s'}$.
But we can **sample**!
Play an entire episode (game) until the end.
$$ S_0, A_0, R_1, S_1, A_1, R_2, \dots, S_T $$

Calculate the return $G_t$ observed.
**Update:** $V(S_t) \approx \text{Average}(G_t)$.

By Law of Large Numbers, Average $\to$ Expectation.

---

## 📝 MC Prediction (Evaluating Policy)

$$ V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)] $$

- $G_t$: Actual return (Target).
- $V(S_t)$: Current estimate.
- $\alpha$: Learning rate.

**Constraint:** Only works for **Episodic** tasks (must terminate).

---

## 🎮 MC Control (Improving Policy)

We need $Q(s, a)$, not just $V(s)$, to pick actions without a model.

1.  **Generate Episode** using $\epsilon$-greedy policy.
2.  **Update Q:** For each $(s, a)$ in episode:
    $$ Q(s, a) \leftarrow Q(s, a) + \alpha [G - Q(s, a)] $$
3.  **Improve Policy:** $\epsilon$-greedy w.r.t new $Q$.

---

## 🆚 MC vs DP

| Feature | Dynamic Programming (DP) | Monte Carlo (MC) |
| :--- | :--- | :--- |
| **Model** | Requires Model $P$ | Model-Free |
| **Update** | Bootstrapping (Uses estimate) | No Bootstrapping (Uses actual return) |
| **Bias/Var** | High Bias, Low Variance | Zero Bias, High Variance |
| **Scope** | All states (Sweep) | Only visited states |

---

## 🎓 Interview Focus

1.  **Why High Variance in MC?**
    - The return $G_t$ depends on *many* random actions and transitions until the end of the episode. One lucky move can change $G_t$ drastically.

2.  **First-Visit vs Every-Visit MC?**
    - **First-Visit:** Only count the *first* time state $s$ is visited in an episode. Unbiased.
    - **Every-Visit:** Count every time. Biased but consistent.

3.  **Off-Policy MC?**
    - Learning about target policy $\pi$ while behaving according to behavior policy $\mu$.
    - Requires **Importance Sampling** ratio $\rho = \frac{\pi(a|s)}{\mu(a|s)}$ to correct the returns. High variance!

---

**Monte Carlo: Learning by doing!**
