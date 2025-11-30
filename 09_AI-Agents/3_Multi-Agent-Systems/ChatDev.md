# ChatDev

> **Virtual Software Company** - Waterfall Model in AI

---

## 🏢 The Simulation

ChatDev (built on Camel/AutoGen concepts) simulates a software company.
It enforces a **Waterfall** process:
1.  **Designing:** CEO + CPO + CTO.
2.  **Coding:** CTO + Programmer + Art Designer.
3.  **Testing:** Programmer + Reviewer + Tester.
4.  **Documenting:** CTO + CEO.

---

## 🗣️ Chat Chains

Instead of a free-for-all group chat, ChatDev uses **Phase-based Chats**.
- **Phase 1:** CEO and CPO discuss features. Output: `Requirements.txt`.
- **Phase 2:** CPO and CTO discuss API. Output: `Design.md`.
- **Phase 3:** CTO and Programmer write code. Output: `main.py`.

---

## 🧠 Thought Instruction

To prevents agents from chatting aimlessly, ChatDev uses **Inception Prompting**.
- "You are the CTO. Your goal is to critique the code. Do not talk about the weather."
- **Self-Reflection:** Agents are asked to summarize the consensus before moving to the next phase.

---

## 🎓 Interview Focus

1.  **Why Waterfall?**
    - For software generation, structure is better than chaos.
    - Defining requirements *before* writing code reduces bugs.

2.  **Hallucination in Files?**
    - ChatDev maintains a virtual file system (Memory).
    - Agents read/write to this shared memory, ensuring the "Tester" sees the exact file the "Programmer" wrote.

---

**ChatDev: Simulating an entire startup in 5 minutes!**
