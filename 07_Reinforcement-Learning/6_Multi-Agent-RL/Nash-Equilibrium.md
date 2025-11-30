# Nash Equilibrium

> **The Stalemate of Intelligence** - Game Theory Basics

---

## ⚖️ The Concept

A set of strategies (policies) is a **Nash Equilibrium** if:
**No player can increase their payoff by changing their strategy unilaterally.**

"Given what everyone else is doing, I am doing the best I can. I have no regret."

---

## 👮 Prisoner's Dilemma

| | Cooperate (Silent) | Defect (Betray) |
| :--- | :--- | :--- |
| **Cooperate** | (-1, -1) | (-3, 0) |
| **Defect** | (0, -3) | **(-2, -2)** |

- If both Cooperate: Total -2. (Global Optimum).
- If I Cooperate, you Defect: I get -3 (Sucker).
- If I Defect, you Cooperate: I get 0 (Free ride).
- **Nash Equilibrium:** Both Defect (-2, -2).
    - If I switch to Cooperate, I get -3 (Worse).
    - Even though (-1, -1) is better for both, rational agents drift to (-2, -2).

---

## ♟️ Minimax (Zero-Sum Games)

In 2-player zero-sum games, Nash Equilibrium is the solution to the **Minimax** problem.
Maximize your minimum possible gain (worst-case scenario).

$$ v = \max_{x} \min_{y} x^T A y $$

- **AlphaZero** approximates this Minimax value using MCTS and Neural Nets.

---

## 🎓 Interview Focus

1.  **Does a Nash Equilibrium always exist?**
    - Yes, in finite games, there is always at least one Nash Equilibrium (possibly involving **Mixed Strategies** - randomizing actions).

2.  **Why is finding Nash Equilibrium hard?**
    - It's PPAD-complete. In high-dimensional spaces (like StarCraft), we can't solve it analytically. We use RL to approximate it.

3.  **Pareto Optimality vs Nash Equilibrium?**
    - **Pareto Optimal:** No one can be made better off without making someone else worse off. (Cooperate, Cooperate) is Pareto Optimal.
    - **Nash:** Stable state. (Defect, Defect) is Nash.
    - Often Nash $\ne$ Pareto (Tragedy of the Commons).

---

**Nash: The balance of power!**
