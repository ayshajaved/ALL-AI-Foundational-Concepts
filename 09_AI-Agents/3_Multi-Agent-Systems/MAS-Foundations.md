# Multi-Agent Systems (MAS) Foundations

> **The Power of Teams** - Collaboration Patterns

---

## 🤝 Why Multiple Agents?

A single agent (like a single human) has limited context, skills, and attention.
**MAS** allows for:
1.  **Specialization:** One agent is the "Coder", another is the "Reviewer".
2.  **Parallelism:** Agents can work on different sub-tasks simultaneously.
3.  **Robustness:** If one agent gets stuck, another can correct it.

---

## 🏗️ Collaboration Patterns

1.  **Sequential (Chain):**
    - Agent A $\to$ Output $\to$ Agent B $\to$ Output.
    - *Example:* Researcher $\to$ Writer $\to$ Editor.

2.  **Hierarchical (Manager-Worker):**
    - **Manager:** Breaks down the goal and assigns tasks.
    - **Workers:** Execute tasks and report back.
    - *Example:* Project Manager assigns tickets to Devs.

3.  **Joint (Group Chat):**
    - All agents share a single context window (chat room).
    - Anyone can speak when it's their turn.
    - Requires a **Router/Moderator** to decide who speaks next.

---

## 🗣️ Communication Protocols

How do agents talk?
- **Natural Language:** "Hey Coder, please fix this bug." (Flexible, but ambiguous).
- **Structured Schema:** JSON messages. (Precise, but rigid).
- **Shared Memory:** Writing to a shared "Blackboard" or Database.

---

## 🎓 Interview Focus

1.  **What is the "Context Window Problem" in MAS?**
    - In a Group Chat, the history grows $N$ times faster (where $N$ is number of agents).
    - **Solution:** Summary agents that compress the history, or private 1-on-1 chats between agents.

2.  **Homogeneous vs Heterogeneous Agents?**
    - **Homogeneous:** Clones of the same agent (e.g., Swarm of drones).
    - **Heterogeneous:** Different prompts/tools (e.g., Coder vs Designer).

---

**MAS: 1 + 1 = 3 (Emergent Intelligence)**
