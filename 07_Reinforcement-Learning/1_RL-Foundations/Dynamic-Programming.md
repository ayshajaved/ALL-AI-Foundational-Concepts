# Dynamic Programming (DP)

> **Solving MDPs when you know the rules** - Policy Iteration vs Value Iteration

---

## 📚 The Setting

We assume we have a **perfect model** of the environment.
We know $P(s'|s,a)$ and $R(s,a,s')$.
This is **Planning**, not Learning.

---

## 1. Policy Evaluation

Given a policy $\pi$, calculate $V_\pi(s)$.
Iteratively update $V(s)$ using Bellman Expectation Equation until convergence.

$$ V_{k+1}(s) \leftarrow \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) [R + \gamma V_k(s')] $$

---

## 2. Policy Iteration

Repeat until convergence:
1.  **Evaluate:** Calculate $V_\pi(s)$ for current $\pi$.
2.  **Improve:** Update $\pi$ to be greedy with respect to $V_\pi$.
    $$ \pi'(s) = \arg\max_a \sum_{s'} P(s'|s,a) [R + \gamma V_\pi(s')] $$

**Guarantee:** Converges to $\pi^*$.

---

## 3. Value Iteration

Combines Evaluation and Improvement into one step.
Iteratively update $V(s)$ using Bellman **Optimality** Equation.

$$ V_{k+1}(s) \leftarrow \max_a \sum_{s'} P(s'|s,a) [R + \gamma V_k(s')] $$

Once $V^*$ converges, extract $\pi^*$.

---

## 💻 Python Implementation (GridWorld)

```python
import numpy as np

def value_iteration(P, R, gamma=0.99, theta=1e-6):
    V = np.zeros(num_states)
    
    while True:
        delta = 0
        for s in range(num_states):
            v = V[s]
            # Calculate value for all actions
            q_values = []
            for a in range(num_actions):
                # Sum over next states
                q = sum([P[s, a, s_next] * (R[s, a, s_next] + gamma * V[s_next]) 
                         for s_next in range(num_states)])
                q_values.append(q)
            
            # Update V(s) to max
            V[s] = max(q_values)
            delta = max(delta, abs(v - V[s]))
            
        if delta < theta:
            break
            
    return V
```

---

## 🎓 Interview Focus

1.  **Policy Iteration vs Value Iteration?**
    - **Policy Iteration:** Fewer iterations, but each iteration is expensive (full evaluation).
    - **Value Iteration:** More iterations, but each iteration is cheap (one sweep). Often faster in practice.

2.  **Curse of Dimensionality?**
    - DP requires iterating over all states. If state space is huge (Go: $10^{170}$), DP is impossible.
    - This is why we need Reinforcement Learning (sampling) and Approximation (Deep Learning).

---

**DP: The classical foundation!**
