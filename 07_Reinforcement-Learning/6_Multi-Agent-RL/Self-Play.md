# Self-Play

> **Beating the Best Version of Yourself** - Curriculum Learning

---

## 🥋 The Idea

In competitive games (Chess, Dota 2), fixed opponents are weak.
To reach superhuman levels, the agent must play against an opponent that is **exactly as strong as itself**.
**Opponent:** A copy of the current policy (or a recent past version).

---

## 📈 The Curriculum

1.  **Beginner:** Random vs Random. (Learns rules).
2.  **Intermediate:** Weak vs Weak. (Learns basic tactics).
3.  **Advanced:** Strong vs Strong. (Learns meta-strategies).
4.  **Expert:** Superhuman vs Superhuman.

The environment difficulty **auto-scales** with the agent's skill.

---

## ⚠️ Fictitious Self-Play (FSP)

**Problem: Cycles.**
Agent A learns Strategy Rock.
Agent B learns Strategy Paper.
Agent A learns Strategy Scissors.
Agent B learns Strategy Rock.
They forget how to beat Paper.

**Solution:**
Don't just play against the *latest* version.
Play against a **mixture of all past versions**.
Ensures the agent is robust against *all* previous strategies, not just the current meta.

---

## 🏆 AlphaStar (StarCraft II)

Used a **League** of agents.
- **Main Agents:** Try to win.
- **Exploiter Agents:** Try to find weaknesses in Main Agents (even if they lose to others).
- **Result:** Robustness against "cheese" strategies.

---

## 🎓 Interview Focus

1.  **Why is Self-Play unstable?**
    - The agent chases a moving target. If the target moves too fast, learning collapses.

2.  **What is Elo Rating?**
    - A metric to track skill in zero-sum games.
    - $P(A \text{ beats } B) = \frac{1}{1 + 10^{(R_B - R_A)/400}}$.

---

**Self-Play: The path to superhuman AI!**
