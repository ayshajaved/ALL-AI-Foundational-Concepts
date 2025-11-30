# AutoGPT & BabyAGI

> **The Birth of Autonomous Agents** - Infinite Loops

---

## 👶 BabyAGI (Task-Driven)

**Core Idea:** An infinite loop of creating and executing tasks.
It uses **Three LLM Calls** per loop:
1.  **Execution Agent:** Do the top task in the list.
2.  **Task Creation Agent:** Based on the result, create new tasks.
3.  **Prioritization Agent:** Re-order the task list.

**Structure:**
- `Task List` (Queue).
- `Vector DB` (Context/Memory).

---

## 🚗 AutoGPT (Goal-Driven)

**Core Idea:** Give it a high-level goal, and let it figure out the rest.
- **Features:**
    - **Internet Access:** Can search the web.
    - **File I/O:** Can write code and save files.
    - **Long-Term Memory:** Pinecone/Chroma.
    - **Self-Correction:** "I tried X and it failed, so I will try Y."

**The Prompt:**
AutoGPT uses a massive system prompt defining its "Constraints", "Resources", and "Performance Evaluation".

---

## 💻 Python Concept (BabyAGI Loop)

```python
task_list = ["Research AI Agents"]

while True:
    if not task_list: break
    
    # 1. Pop Task
    current_task = task_list.pop(0)
    
    # 2. Execute
    result = execute_task(current_task)
    print(f"Result: {result}")
    
    # 3. Create New Tasks
    new_tasks = task_creation_agent(result, current_task, task_list)
    task_list.extend(new_tasks)
    
    # 4. Prioritize
    task_list = prioritization_agent(task_list)
```

---

## 🎓 Interview Focus

1.  **Why do these agents get stuck?**
    - **Loops:** "I need to search Google" $\to$ "I need to search Google".
    - **Context Overflow:** The history gets too long, and important details are dropped.
    - **Hallucination:** The agent invents a file that doesn't exist and tries to read it.

2.  **Difference from ReAct?**
    - ReAct is a *reasoning pattern* (Thought-Action).
    - AutoGPT/BabyAGI are *architectures* (Task Queues, Memory Management) that *use* reasoning patterns.

---

**AutoGPT: The viral experiment that started it all!**
