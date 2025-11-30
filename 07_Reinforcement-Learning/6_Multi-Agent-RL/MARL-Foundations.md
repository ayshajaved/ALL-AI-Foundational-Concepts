# Multi-Agent RL (MARL) Foundations

> **Beyond the Solitary Agent** - Cooperation, Competition, and Chaos

---

## 👥 The Setting

Single-Agent RL: Agent vs Static Environment.
Multi-Agent RL: Agent vs Other Agents vs Environment.

**The Environment is Non-Stationary:**
From the perspective of Agent 1, the environment is changing because Agent 2 is learning and changing its policy.
"I learned to block left because Agent 2 shoots left. But now Agent 2 learned to shoot right. My policy is now bad."

---

## 🎭 Types of Games

1.  **Cooperative (Team):**
    - All agents share the same reward. $R_1 = R_2 = \dots = R$.
    - Goal: Maximize common return.
    - *Example:* Robots moving a heavy table together.

2.  **Competitive (Zero-Sum):**
    - $R_1 = -R_2$.
    - Goal: I win, you lose.
    - *Example:* Chess, Tennis.

3.  **Mixed-Sum (General Sum):**
    - Agents have different goals, but might benefit from cooperation.
    - *Example:* Prisoner's Dilemma, Self-Driving Cars at an intersection.

---

## 🧩 Centralized vs Decentralized

1.  **Centralized Training, Centralized Execution (CTCE):**
    - One giant brain controls all bodies.
    - *Problem:* Action space explodes ($|A|^N$). Communication constraints.

2.  **Decentralized Training, Decentralized Execution (DTDE):**
    - Each agent learns on its own (Independent RL).
    - *Problem:* Non-stationarity.

3.  **Centralized Training, Decentralized Execution (CTDE):**
    - **Training:** Agents can see everything (global state, other agents' actions).
    - **Execution:** Agents act based only on their local observations.
    - *Standard for MARL (e.g., MADDPG).*

---

## 🎓 Interview Focus

1.  **Why is MARL harder than Single-Agent RL?**
    - **Non-Stationarity:** The transition dynamics $P(s'|s, a_1, \dots, a_N)$ depend on others.
    - **Scalability:** Joint action space grows exponentially.
    - **Credit Assignment:** Who contributed to the team win?

2.  **What is the "Lazy Agent" problem?**
    - In cooperative games, one agent might learn to do nothing if the other agent is good enough to get the reward alone.

---

**MARL: It takes a village to raise an AI!**
