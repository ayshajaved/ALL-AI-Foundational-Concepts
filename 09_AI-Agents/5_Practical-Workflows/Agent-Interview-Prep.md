# Agent Interview Prep

> **Mastering the Agent Interview** - Top Questions

---

## 🟢 Beginner (Concepts)

1.  **What is the difference between a Chatbot and an Agent?** Chatbot = Passive (Input/Output). Agent = Active (Reasoning/Action/Loop).
2.  **Explain ReAct.** Reason + Act. The loop of Thought $\to$ Action $\to$ Observation.
3.  **What is a System Prompt?** The initial instruction that defines the agent's persona, constraints, and tools.
4.  **What is Function Calling?** A structured way for LLMs to request the execution of code/APIs by outputting JSON.
5.  **Why do agents need memory?** LLMs are stateless. Memory allows them to maintain context over long tasks.

---

## 🟡 Intermediate (Architectures)

6.  **Explain the AutoGPT loop.** Task Queue $\to$ Execute $\to$ Create New Tasks $\to$ Prioritize.
7.  **What is RAG vs Tool Use?** RAG is for retrieving information (Read). Tool Use is for taking action (Write/Execute). Agents use both.
8.  **How does CrewAI handle delegation?** It uses a hierarchical process where a Manager agent breaks down tasks and assigns them to worker agents based on roles.
9.  **What is "Reflexion"?** A self-correction loop where the agent evaluates its own failure and generates a "verbal reinforcement" to avoid the mistake next time.
10. **What is the risk of infinite loops?** The agent keeps trying the same failed action. Fix: Max iterations, Timeout, or Human interrupt.

---

## 🔴 Advanced (System Design)

11. **Design a Coding Agent.**
    - **State:** File system, Git history.
    - **Tools:** `read`, `write`, `lint`, `test`.
    - **Safety:** Docker sandbox.
    - **Loop:** Write $\to$ Lint $\to$ Fix $\to$ Test $\to$ Submit.

12. **Design a Travel Planning Agent.**
    - **Tools:** Flight Search, Hotel Search, Calendar.
    - **Planning:** Decomposition ("Book flight", "Book hotel").
    - **Constraint Satisfaction:** "Hotel must be < $200 AND near airport."
    - **Memory:** User preferences (Window seat).

13. **How to evaluate a Customer Support Agent?**
    - **Success Rate:** Did the user say "Thanks"?
    - **Escalation Rate:** How often did it fail and call a human?
    - **Safety:** Did it promise a refund it couldn't give? (Tool permissions).

---

**You are now an Agentic AI Expert!**
