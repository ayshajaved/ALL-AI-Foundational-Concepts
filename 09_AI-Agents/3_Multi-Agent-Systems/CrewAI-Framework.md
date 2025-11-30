# CrewAI Framework

> **Role-Playing Agents** - Orchestrating Teams

---

## 🎭 The Philosophy

CrewAI is built on top of LangChain.
It emphasizes **Role-Playing**.
- You don't just give an agent a tool.
- You give it a **Backstory**, a **Goal**, and a **Role**.

---

## 🧱 Core Components

1.  **Agent:**
    - `Role`: "Senior Python Engineer".
    - `Goal`: "Write clean, efficient code".
    - `Backstory`: "You are a veteran engineer at Google..."
    - `Tools`: [FileRead, GitHubSearch].

2.  **Task:**
    - `Description`: "Create a snake game".
    - `Expected Output`: "A python file named snake.py".
    - `Agent`: Assigned to the Coder.

3.  **Crew:**
    - The container that manages the agents and tasks.
    - **Process:** `Sequential` (default) or `Hierarchical`.

---

## 💻 Implementation

```python
from crewai import Agent, Task, Crew, Process

# 1. Define Agents
researcher = Agent(
    role='Researcher',
    goal='Discover new AI trends',
    backstory='You are a curious analyst.',
    verbose=True
)

writer = Agent(
    role='Tech Writer',
    goal='Write engaging blog posts',
    backstory='You simplify complex topics.',
    verbose=True
)

# 2. Define Tasks
task1 = Task(description='Find news about AI Agents', agent=researcher)
task2 = Task(description='Write a blog post based on the news', agent=writer)

# 3. Create Crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    process=Process.sequential
)

# 4. Kickoff
result = crew.kickoff()
print(result)
```

---

## 🎓 Interview Focus

1.  **Why Backstory?**
    - It acts as a "System Prompt" that steers the style and behavior of the LLM.
    - A "Grumpy Reviewer" finds more bugs than a "Nice Reviewer".

2.  **Hierarchical Process?**
    - CrewAI automatically creates a "Manager Agent" (using GPT-4) that delegates tasks to the others and reviews their work.

---

**CrewAI: Making agents feel like coworkers!**
