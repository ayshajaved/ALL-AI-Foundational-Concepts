# Rainbow DQN

> **The Avengers of RL** - Combining 7 Improvements

---

## 🌈 The 7 Components

Hessel et al. (DeepMind, 2017) combined 7 independent improvements into one state-of-the-art agent.

1.  **DQN:** The base algorithm.
2.  **Double DQN:** Reduces overestimation bias.
3.  **Prioritized Experience Replay (PER):** Learns from hard examples.
4.  **Dueling Networks:** Separates Value and Advantage.
5.  **Multi-Step Learning (N-step TD):**
    - Look $n$ steps ahead.
    - $R_t + \gamma R_{t+1} + \dots + \gamma^n \max Q(S_{t+n})$.
    - Faster propagation of rewards.
6.  **Distributional RL (C51):**
    - Instead of predicting the *mean* Q-value, predict the **distribution** of returns (probability mass function).
    - "This action has a 10% chance of 0 reward and 90% chance of 100 reward."
7.  **Noisy Nets:**
    - Replaces $\epsilon$-greedy exploration.
    - Adds learnable Gaussian noise to the weights of the Linear layers.
    - The network *learns* when to explore (high noise) and when to exploit (low noise).

---

## 📊 Results

Rainbow significantly outperformed all individual components on the Atari benchmark.
It proved that these improvements are largely **complementary**.

---

## 💻 Distributional RL (Concept)

Standard Q-Learning:
$$ Q(s, a) \leftarrow \mathbb{E}[R + \gamma Q(s', a')] $$

Distributional Q-Learning:
$$ Z(s, a) \stackrel{D}{=} R + \gamma Z(s', a') $$
- $Z$ is a random variable.
- We minimize the **KL Divergence** (or Wasserstein distance) between the predicted distribution and the target distribution.

---

## 🎓 Interview Focus

1.  **Why is Distributional RL better?**
    - It captures **risk**. A mean of 50 could be "Always 50" or "50% chance of 0, 50% chance of 100".
    - It helps with learning dynamics (richer signal).

2.  **What is the cost of Rainbow?**
    - Complexity. Implementing all 7 parts is difficult and prone to bugs.
    - Computational cost (Distributional RL requires predicting 51 atoms instead of 1 scalar).

---

**Rainbow: The ultimate Value-Based agent!**
