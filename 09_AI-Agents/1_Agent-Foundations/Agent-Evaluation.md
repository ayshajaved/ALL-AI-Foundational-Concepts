# Agent Evaluation

> **Grading the Brain** - How do we know it works?

---

## 📉 The Difficulty

Evaluating a chatbot is hard (BLEU/ROUGE are useless).
Evaluating an agent is harder.
- It might take a different path to the same solution.
- It might get stuck in a loop.
- It might use tools incorrectly but get the right answer by luck.

---

## 🧪 Evaluation Frameworks

1.  **AgentBench:**
    - A benchmark of 8 environments (OS, Database, Knowledge Graph, etc.).
    - Measures **Success Rate** (Did it achieve the goal?).

2.  **Trajectory Evaluation:**
    - Don't just check the answer. Check the **Process**.
    - Did it search for "Apple" when asked about "Banana"?
    - **LLM-as-a-Judge:** Use GPT-4 to grade the trace of a smaller agent.

3.  **Unit Testing for Agents:**
    - **Mocking:** Mock the tools.
    - **Deterministic Tests:** "Given this exact tool output, does the agent output this exact thought?"

---

## 📊 Metrics

- **Success Rate (SR):** % of tasks completed.
- **Steps to Solution:** Efficiency. Lower is better.
- **Tool Error Rate:** How often did it generate invalid JSON or call non-existent tools?
- **Cost:** Tokens consumed per task.

---

## 💻 Example: LLM-as-a-Judge Prompt

```text
You are an expert evaluator.
Review the following Agent Trajectory.

Goal: "Find the price of BTC."

Trajectory:
1. Action: Search("Weather in NY") -> Obs: "20C"
2. Thought: "I need to find BTC price."
3. Action: Search("BTC Price") -> Obs: "$50,000"
4. Final Answer: "$50,000"

Score (1-5): 3
Reasoning: The agent eventually got the answer, but Step 1 was irrelevant and wasteful.
```

---

## 🎓 Interview Focus

1.  **Why is "Success Rate" insufficient?**
    - An agent might delete the database to "solve" the problem of a slow query.
    - We need **Safety** and **Side-effect** evaluation.

2.  **How to debug an agent loop?**
    - **Tracing:** Use tools like **LangSmith** or **Arize Phoenix**.
    - Visualize the tree of calls. See exactly where the prompt failed or the tool returned an error.

---

**Evaluation: Moving from "It looks cool" to "It works"!**
