# LangChain Agents

> **The Orchestrator** - Chains, Tools, and Runtimes

---

## 🔗 The Concept

LangChain provides the "glue" to build agents.
It standardizes:
1.  **Prompt Templates:** Managing the ReAct/CoT prompts.
2.  **Tool Interfaces:** A common wrapper for Google Search, Calculator, etc.
3.  **Output Parsers:** Extracting `Action` and `Action Input` from text.

---

## 🤖 Agent Types

1.  **Zero-Shot ReAct:**
    - Uses the standard "Thought/Action/Observation" loop.
    - Works with any model (even non-fine-tuned ones).
    - *Prompt:* "Use the following tools..."

2.  **OpenAI Functions Agent:**
    - Optimized for GPT-3.5/4.
    - Uses `tool_calls` API instead of text parsing.
    - More robust.

3.  **Plan-and-Execute:**
    - Step 1: Planner LLM generates a list of steps.
    - Step 2: Executor Agent runs the steps one by one.

---

## 💻 Implementation (OpenAI Functions)

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# 1. Define Tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return a * b

tools = [multiply]

# 2. Model & Prompt
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# 3. Create Agent
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 4. Run
agent_executor.invoke({"input": "What is 5 times 8?"})
```

---

## 🎓 Interview Focus

1.  **AgentExecutor vs LCEL (LangChain Expression Language)?**
    - `AgentExecutor` is the legacy runtime (loop).
    - Modern LangChain uses `LangGraph` (built on LCEL) to define agents as state machines (Cyclic Graphs).

2.  **Memory in Agents?**
    - The agent needs to remember previous steps.
    - LangChain passes `agent_scratchpad` (a list of intermediate steps) into the prompt every time the loop runs.

---

**LangChain: The standard library for Agentic AI!**
