# Planning Algorithms

> **Thinking Ahead** - Decomposition and Self-Correction

---

## 🗺️ Why Planning?

For simple tasks ("What is the capital of France?"), direct execution works.
For complex tasks ("Write a video game"), the agent needs a **Plan**.
Without a plan, the agent gets lost in the details or goes in circles.

---

## 🧩 Decomposition

Breaking a complex goal into sub-goals.
**Prompt:**
"Goal: Write a Snake game.
Plan:
1. Create the game window (pygame).
2. Handle user input (arrow keys).
3. Implement snake movement logic.
4. Implement food and scoring.
5. Game over logic."

**Execution:** The agent then executes Step 1, then Step 2, etc.

---

## 🔄 Self-Correction (Reflexion)

Agents make mistakes.
**Reflexion** (Shinn et al., 2023) adds a feedback loop.
1.  **Actor:** Generates a trajectory (Attempt 1).
2.  **Evaluator:** Scores the trajectory. "Failed. The snake goes through walls."
3.  **Self-Reflection:** "I failed because I didn't check for collision with window boundaries."
4.  **Actor:** Generates Attempt 2, conditioned on the Reflection.

---

## 🌳 Tree of Thoughts (ToT) Search

Instead of just one plan, generate 3 possible plans.
- **BFS (Breadth-First Search):** Explore all 3 plans for the first step. Keep the best ones.
- **DFS (Depth-First Search):** Go deep on one plan. Backtrack if it fails.

---

## 💻 Python Concept (Reflexion Loop)

```python
def solve_task(task, max_retries=3):
    memory = []
    
    for i in range(max_retries):
        # 1. Act
        solution = agent.act(task, context=memory)
        
        # 2. Evaluate
        success, feedback = environment.evaluate(solution)
        
        if success:
            return solution
            
        # 3. Reflect
        reflection = agent.reflect(task, solution, feedback)
        memory.append(f"Attempt {i}: Failed. Feedback: {feedback}. Insight: {reflection}")
        
    return "Failed"
```

---

## 🎓 Interview Focus

1.  **Plan-and-Solve vs ReAct?**
    - **Plan-and-Solve:** Plan once, execute all. (Good for known tasks).
    - **ReAct:** Plan one step, execute, observe, re-plan. (Good for unknown/dynamic environments).

2.  **What is "Hindsight Experience Replay" in Agents?**
    - Storing failed trajectories in memory but relabeling them as "How NOT to do X", so the agent learns from failure.

---

**Planning: The pre-frontal cortex of the Agent!**
