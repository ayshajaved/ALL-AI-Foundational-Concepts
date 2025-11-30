# RL Interview Prep: Top 50 Q&A

> **Mastering the RL Interview** - From Bellman to AlphaZero

---

## 🟢 Beginner (Concepts)

1.  **What is the Markov Property?** Future depends only on present, not past.
2.  **Exploration vs Exploitation?** Gathering info vs Maximizing reward. $\epsilon$-greedy.
3.  **On-Policy vs Off-Policy?** Learning the policy you act with (SARSA) vs Learning the optimal policy while acting differently (Q-Learning).
4.  **What is a Discount Factor?** Values immediate rewards over future ones. Ensures convergence.
5.  **Explain the Bellman Equation.** Recursive definition of Value. $V = R + \gamma V'$.

---

## 🟡 Intermediate (Algorithms)

6.  **Why Experience Replay in DQN?** Breaks correlation between samples. Stabilizes training.
7.  **Why Target Network in DQN?** Fixes the moving target problem.
8.  **Policy Gradient vs Q-Learning?** PG learns prob distribution (good for continuous). Q-Learning learns values (sample efficient).
9.  **What is the Advantage Function?** $A(s, a) = Q(s, a) - V(s)$. How much better is action $a$ than average?
10. **Explain PPO Clipping.** Prevents policy from changing too much. Trust Region.

---

## 🔴 Advanced (Modern RL)

11. **Why is Model-Based RL sample efficient?** It learns the dynamics, allowing planning (simulation) without real interaction.
12. **Explain AlphaZero's Loss.** MSE for Value + CrossEntropy for Policy (matching MCTS counts).
13. **What is Distributional RL?** Learning the full distribution of returns, not just the mean.
14. **Why Entropy Regularization (SAC)?** Encourages exploration and robustness.
15. **What is Hindsight Experience Replay (HER)?** Pretending a failed goal was actually the goal we wanted to achieve. Learns from failure.
16. **Explain GAE.** Generalized Advantage Estimation. Balances bias (TD) and variance (MC) using $\lambda$.
17. **Nash Equilibrium in MARL?** State where no agent benefits from changing strategy unilaterally.
18. **Why is Offline RL hard?** Distribution shift. The agent queries Q-values for unseen actions.
19. **What is Curriculum Learning?** Training on easy tasks first, then hard ones (Self-Play).
20. **Vanishing Gradients in RL?** Long time horizons. Solved by GAE and proper credit assignment.

---

## 🧠 System Design Scenarios

**Q: Design an RL agent for Traffic Light Control.**
- **State:** Queue lengths, waiting times.
- **Action:** Switch Light (Red/Green).
- **Reward:** -1 * Total Waiting Time.
- **Algorithm:** PPO (Continuous monitoring) or DQN (Discrete switching).
- **Sim:** SUMO Simulator.

**Q: Design a Recommendation System using RL.**
- **State:** User history, Context.
- **Action:** Recommend Item X.
- **Reward:** Click (+1), Purchase (+10), Skip (-1).
- **Algorithm:** Contextual Bandits (Simplified RL) or DDPG (Large action space embedding).
- **Off-Policy:** Must learn from logged data (Offline RL).

---

**You are now an RL Expert. Go solve intelligence!**
