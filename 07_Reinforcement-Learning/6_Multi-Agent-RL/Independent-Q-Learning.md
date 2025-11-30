# Independent Q-Learning (IQL)

> **Ignoring the Elephant in the Room** - Naive MARL

---

## 🙈 The Approach

Treat other agents as part of the **Environment**.
Agent $i$ learns $Q_i(s, a_i)$.
It ignores the actions of agents $j \ne i$.

$$ L(\theta_i) = (r_i + \gamma \max_{a_i'} Q_i(s', a_i'; \theta_i^-) - Q_i(s, a_i; \theta_i))^2 $$

---

## 📉 The Problem: Non-Stationarity

Since other agents are learning, the environment dynamics change.
- Episode 1: Agent 2 moves Randomly. I learn to exploit it.
- Episode 100: Agent 2 becomes Smart. My policy fails.
- The "Target" is moving. Experience Replay becomes dangerous (old data reflects old opponent policies).

---

## 🚀 Why use it?

Despite the theoretical flaws, IQL works surprisingly well in practice (e.g., Overcooked, simple swarms).
**Pros:**
- Scalable (Linear with N agents).
- Decentralized.

**Cons:**
- Fails in coordination tasks requiring precise synchronization.
- Oscillates (Rock-Paper-Scissors loop).

---

## 🎓 Interview Focus

1.  **How to fix Experience Replay in IQL?**
    - **Fingerprinting:** Add the iteration number or opponent's policy ID to the state. $Q(s, \text{step}, a)$.
    - **Lenient Experience Replay:** Only overwrite low rewards, keep high rewards (optimism).

2.  **Is IQL guaranteed to converge?**
    - No. Only in static environments. In MARL, it often enters limit cycles.

---

**IQL: Simple, naive, but effective!**
