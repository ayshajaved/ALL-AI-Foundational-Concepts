# Prompting for Agents

> **How to Think** - ReAct, CoT, and Tree of Thoughts

---

## 💭 Chain of Thought (CoT)

Standard Prompting: `Q: What is 2+2? A: 4`
CoT Prompting: `Q: What is 2+2? A: Let's think step by step. 2 plus 2 is 4. The answer is 4.`

**Why it matters:**
- LLMs are autoregressive. Writing out the reasoning *before* the answer allows the model to compute intermediate states.
- It reduces hallucination and improves math/logic performance.

---

## 🔄 ReAct (Reason + Act)

The gold standard for Agents (Yao et al., 2022).
It forces the model to interleave **Thought**, **Action**, and **Observation**.

**Prompt Template:**
```text
You are an agent.
To solve a problem, use the following format:

Thought: Analyze the current situation.
Action: Choose a tool to use [Search, Calculator].
Action Input: The input for the tool.
Observation: The result of the tool (provided by system).
... (Repeat) ...
Final Answer: The final response to the user.
```

**Example Trace:**
> **Thought:** I need to find the president of France and his age.
> **Action:** Search
> **Action Input:** "President of France 2024"
> **Observation:** Emmanuel Macron.
> **Thought:** Now I need his age.
> **Action:** Search
> **Action Input:** "Emmanuel Macron age"
> **Observation:** 46 years old.
> **Final Answer:** Emmanuel Macron is 46.

---

## 🌳 Tree of Thoughts (ToT)

For complex planning, linear thinking isn't enough.
**ToT** explores multiple branches of reasoning.
1.  **Generate:** Propose 3 possible next steps.
2.  **Evaluate:** Score each step (Is this promising?).
3.  **Select:** Keep the best step (BFS/DFS search).

---

## 💻 Implementation (ReAct Prompt)

```python
REACT_PROMPT = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""
```

---

## 🎓 Interview Focus

1.  **Why does ReAct fail?**
    - **Looping:** The agent gets stuck doing the same search over and over.
    - **Context Limit:** The history (Thought/Obs/Thought/Obs) grows too long for the context window.

2.  **Plan-and-Solve vs ReAct?**
    - **ReAct:** Think-Act-Think-Act (Step-by-step). Good for dynamic environments.
    - **Plan-and-Solve:** Generate a full plan first (1. Search X, 2. Calc Y), then execute. Good for static tasks.

---

**Prompting: The programming language of Agents!**
