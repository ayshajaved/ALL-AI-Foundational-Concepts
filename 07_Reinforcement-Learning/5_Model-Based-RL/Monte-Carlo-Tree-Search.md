# Monte Carlo Tree Search (MCTS)

> **The Engine of AlphaGo** - Decision Time Planning

---

## 🌳 The Idea

Instead of learning a policy for *every* state (which is hard), just plan for the *current* state when you are there.
Build a search tree of possibilities.

---

## ⚙️ The 4 Phases

1.  **Selection:**
    Start from root. Traverse down the tree using a **Tree Policy** (e.g., UCT) until a leaf node is reached.
    
2.  **Expansion:**
    Add a new child node to the leaf (if not terminal).
    
3.  **Simulation (Rollout):**
    Play a random game from the new node to the end.
    Result: Win (+1) or Loss (0).
    
4.  **Backpropagation:**
    Update the value and visit count of all nodes along the path.
    $V(s) \leftarrow V(s) + \text{Result}$.
    $N(s) \leftarrow N(s) + 1$.

---

## ⚖️ UCT (Upper Confidence Bound for Trees)

How to select the next node?
Balance **Exploitation** (high win rate) and **Exploration** (low visit count).

$$ \text{UCT} = \frac{W_i}{N_i} + C \sqrt{\frac{\ln N_{parent}}{N_i}} $$

- $W_i/N_i$: Average win rate (Exploit).
- $\sqrt{\dots}$: Exploration bonus (Explores rarely visited nodes).
- $C$: Exploration constant ($\sqrt{2}$).

---

## 🎓 Interview Focus

1.  **Why MCTS instead of Minimax?**
    - Minimax requires an evaluation function (heuristic) at the depth limit. In Go, it's hard to write a heuristic.
    - MCTS uses **rollouts** (random play) to estimate value, which requires no domain knowledge (except rules).

2.  **Does MCTS learn?**
    - Standard MCTS does not "learn" a policy that persists. The tree is discarded after the move.
    - **AlphaZero** combines MCTS with a Neural Network to learn a persistent policy.

---

**MCTS: Thinking ahead, one branch at a time!**
