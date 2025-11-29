# Bellman Equations

> **The Recursive Magic** - Relating Value to Next Value

---

## 💎 Value Functions

How good is it to be in state $s$?

1.  **State-Value Function $V_\pi(s)$:**
    Expected return starting from $s$ and following policy $\pi$.
    $$ V_\pi(s) = \mathbb{E}_\pi [G_t | S_t = s] $$

2.  **Action-Value Function $Q_\pi(s, a)$:**
    Expected return starting from $s$, taking action $a$, and *then* following $\pi$.
    $$ Q_\pi(s, a) = \mathbb{E}_\pi [G_t | S_t = s, A_t = a] $$

---

## 🔄 The Bellman Expectation Equation

Decomposes Value into: **Immediate Reward + Discounted Value of Next State**.

$$ V_\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V_\pi(s')] $$

In English:
"Value of $s$ = Average over actions [ Reward + $\gamma$ * Value of next state ]"

Similarly for Q:
$$ Q_\pi(s, a) = \sum_{s'} P(s'|s,a) [R + \gamma \sum_{a'} \pi(a'|s') Q_\pi(s', a')] $$

---

## 🌟 The Bellman Optimality Equation

The value of the *best* policy $\pi^*$.
Instead of averaging over actions, we take the **max**.

$$ V^*(s) = \max_a \sum_{s'} P(s'|s,a) [R + \gamma V^*(s')] $$

$$ Q^*(s, a) = \sum_{s'} P(s'|s,a) [R + \gamma \max_{a'} Q^*(s', a')] $$

**Key Insight:** If we know $Q^*(s, a)$, the optimal policy is simply to be greedy:
$$ \pi^*(s) = \arg\max_a Q^*(s, a) $$

---

## 🎓 Interview Focus

1.  **Why is the Bellman Equation important?**
    - It turns the optimization problem into a set of linear equations (for small MDPs) or an iterative update rule (for large MDPs/RL).
    - Most RL algorithms (Q-Learning, SARSA) are just applying the Bellman update operator.

2.  **Relationship between $V$ and $Q$?**
    - $V^*(s) = \max_a Q^*(s, a)$.
    - The value of a state is the value of the best action you can take in that state.

---

**Bellman: The recursive engine of RL!**
