# OpenAI Assistants API

> **Agents as a Service** - Threads, Runs, and Tools

---

## ☁️ The Managed Service

OpenAI handles the state, memory, and tool execution on their servers.
You don't need a Vector DB or a Python loop.

**Key Concepts:**
1.  **Assistant:** The agent definition (Model, Instructions, Tools).
2.  **Thread:** The conversation history (State). Infinite context window (managed by OpenAI).
3.  **Run:** The execution of an Assistant on a Thread.
4.  **Run Step:** Individual actions (Tool calls, Message creation).

---

## 🛠️ Built-in Tools

1.  **Code Interpreter:**
    - A sandboxed Python environment.
    - Can generate charts, process CSVs, solve math.
    - *Crucial:* It can write code to solve problems the LLM is bad at (math).
2.  **File Search (Retrieval):**
    - Upload PDFs. OpenAI chunks, embeds, and searches them automatically.
3.  **Function Calling:**
    - You define the schema, OpenAI tells you to run it.

---

## 💻 Implementation

```python
from openai import OpenAI

client = OpenAI()

# 1. Create Assistant
assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a personal math tutor. Write and run code to answer math questions.",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4-turbo"
)

# 2. Create Thread
thread = client.beta.threads.create()

# 3. Add Message
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Solve 3x + 11 = 14"
)

# 4. Run
run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id,
    assistant_id=assistant.id,
)

# 5. Get Response
if run.status == 'completed': 
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    print(messages.data[0].content[0].text.value)
```

---

## 🎓 Interview Focus

1.  **Pros vs Cons?**
    - **Pros:** extremely easy. No infrastructure. State management is handled.
    - **Cons:** Vendor lock-in. Data privacy (files on OpenAI servers). Cost (can be expensive).

2.  **How does Code Interpreter work?**
    - It generates Python code, executes it in a Jupyter kernel, and reads the `stdout`/`stderr`.
    - It can loop: Write code $\to$ Error $\to$ Rewrite code $\to$ Success.

---

**Assistants API: The fastest way to build powerful agents!**
