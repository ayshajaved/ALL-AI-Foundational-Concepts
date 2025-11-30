# Local Agent Stack

> **Privacy First** - Ollama + Local Tools

---

## 🔒 Why Local?

- **Privacy:** Don't send PII/Code to OpenAI.
- **Cost:** Free (after hardware).
- **Offline:** Works on a plane.

---

## 🛠️ The Stack

- **Model:** `Llama 3` or `Mistral` (via **Ollama**).
- **Inference:** `Ollama` provides a local API (`localhost:11434`).
- **Framework:** `LangChain` (supports Ollama).

---

## 💻 Implementation

```python
from langchain_community.chat_models import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# 1. Local LLM
llm = ChatOllama(model="llama3")

# 2. Define Tool
@tool
def get_system_time(query: str) -> str:
    """Returns the current system time."""
    import datetime
    return str(datetime.datetime.now())

tools = [get_system_time]

# 3. Prompt (Download standard ReAct prompt)
prompt = hub.pull("hwchase17/react")

# 4. Create Agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. Run
agent_executor.invoke({"input": "What time is it?"})
```

---

## 🎓 Interview Focus

1.  **Function Calling on Local Models?**
    - Llama 3 is decent at it. Older models (Llama 2) struggled.
    - **Gorilla LLM:** A model fine-tuned specifically for API calling.
    - **Grammars:** Use `llama.cpp` grammars to force valid JSON output from local models.

2.  **Latency?**
    - Local inference can be slow on CPU.
    - **Quantization:** Use 4-bit (GGUF) models to fit in RAM and run faster.

---

**Local Stack: Your own private JARVIS!**
