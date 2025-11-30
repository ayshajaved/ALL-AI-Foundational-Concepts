# Microsoft AutoGen

> **Conversational Agents** - The Group Chat Manager

---

## 🗣️ The Paradigm

AutoGen treats everything as a **Conversation**.
- Coding? A conversation between a User and an Assistant.
- Debugging? A conversation between a Coder and an Executor.

**Key Feature:** **Code Execution**.
AutoGen agents run code locally (Docker) to solve tasks, rather than just simulating it.

---

## 🏗️ Architecture

1.  **ConversableAgent:** The base class. Can send/receive messages.
2.  **UserProxyAgent:**
    - Represents the Human.
    - Can execute code blocks detected in the received message.
    - Can prompt the human for input.
3.  **AssistantAgent:**
    - The LLM (GPT-4).
    - Generates code/plans.
4.  **GroupChatManager:**
    - Manages a chat with >2 agents.
    - Selects the next speaker (Round Robin, Random, or LLM-based selection).

---

## 💻 Implementation (Two-Agent Coding)

```python
from autogen import UserProxyAgent, AssistantAgent

# 1. Assistant (The Brain)
assistant = AssistantAgent(
    name="coder",
    llm_config={"config_list": config_list}
)

# 2. User Proxy (The Executor)
user_proxy = UserProxyAgent(
    name="user_proxy",
    code_execution_config={"work_dir": "coding"},
    human_input_mode="NEVER" # Fully autonomous
)

# 3. Initiate Chat
user_proxy.initiate_chat(
    assistant,
    message="Plot a chart of NVDA stock price YTD."
)
```

**What happens:**
1.  `coder` writes Python code to fetch stock data.
2.  `user_proxy` executes the code.
3.  If error, `user_proxy` sends the traceback to `coder`.
4.  `coder` fixes the code.
5.  Loop continues until success.

---

## 🎓 Interview Focus

1.  **Human-in-the-loop?**
    - `human_input_mode="ALWAYS"`: The agent asks you before every step.
    - `human_input_mode="TERMINATE"`: The agent runs autonomously until it thinks it's done, then asks for feedback.

2.  **Why is AutoGen powerful for coding?**
    - The feedback loop (Code $\to$ Error $\to$ Fix) is native to the framework.

---

**AutoGen: The framework for self-healing code!**
