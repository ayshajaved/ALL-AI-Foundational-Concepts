# AlphaZero Deep Dive

> **Tabula Rasa** - Mastering Games without Human Knowledge

---

## 🧠 The Architecture

AlphaZero replaces the random rollout of MCTS with a Neural Network.
$$ f_\theta(s) \to (\mathbf{p}, v) $$
- **$\mathbf{p}$ (Policy):** Vector of move probabilities.
- **$v$ (Value):** Scalar evaluation of the position ($-1$ to $+1$).

---

## 🔄 The Loop (Self-Play)

1.  **MCTS Search:**
    - Use the network $f_\theta$ to guide the MCTS selection (instead of UCT).
    - The MCTS produces a refined policy $\pi_{mcts}$ (visit counts).
    - $\pi_{mcts}$ is *better* than the raw network output $\mathbf{p}$.

2.  **Play:**
    - Agent plays against itself using $\pi_{mcts}$.
    - Generate data: $(s, \pi_{mcts}, z)$, where $z$ is the final game outcome.

3.  **Train Network:**
    - Train $f_\theta$ to predict the MCTS policy and the game outcome.
    - Loss: $L = (z - v)^2 - \pi_{mcts}^T \log \mathbf{p} + c ||\theta||^2$.
    - (MSE for Value + CrossEntropy for Policy).

---

## 🚀 Why is it revolutionary?

- **No Human Data:** Starts from random play.
- **General:** Works for Chess, Go, Shogi, Atari (MuZero).
- **Policy Improvement:** The MCTS acts as a **Policy Improvement Operator**. The network learns to imitate the "smarter" search.

---

## 🎓 Interview Focus

1.  **AlphaGo vs AlphaZero?**
    - **AlphaGo:** Trained on human expert games (Supervised) + RL. Used hand-crafted features.
    - **AlphaZero:** Pure RL (Self-play). No human data. Raw board input.

2.  **What is MuZero?**
    - AlphaZero requires a perfect simulator (rules of the game).
    - **MuZero** learns the rules (dynamics model) inside a latent space, allowing it to master Atari games where rules are unknown.

---

**AlphaZero: The God of Board Games!**
