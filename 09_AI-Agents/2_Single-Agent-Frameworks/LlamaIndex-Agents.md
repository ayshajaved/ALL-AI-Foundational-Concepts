# LlamaIndex Agents

> **Data-Centric Agents** - RAG + Reasoning

---

## 📚 The Focus

LangChain focuses on "Chains" and "Tools".
LlamaIndex focuses on **Data**.
LlamaIndex Agents are designed to perform complex **Reasoning over Data** (RAG).

---

## 🔍 Query Engine Tools

In LlamaIndex, a "Tool" is often a **Query Engine** wrapping a Vector Store.
- **Tool A:** Search the "Q1 Financial Report".
- **Tool B:** Search the "Q2 Financial Report".
- **Agent:** "Compare the revenue growth between Q1 and Q2."
    - Action 1: Call Tool A.
    - Action 2: Call Tool B.
    - Action 3: Compute difference.

---

## 🏗️ ReAct vs OpenAIAgent

LlamaIndex provides specialized workers:
1.  **ReActAgent:** Works with any LLM.
2.  **OpenAIAgent:** Uses Function Calling (State-of-the-Art performance).

---

## 💻 Implementation

```python
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent

# 1. Wrap RAG engines as tools
query_engine_tools = [
    QueryEngineTool(
        query_engine=sept_engine,
        metadata=ToolMetadata(
            name="sept_22",
            description="Provides information about Uber quarterly financials ending September 2022",
        ),
    ),
    QueryEngineTool(
        query_engine=june_engine,
        metadata=ToolMetadata(
            name="june_22",
            description="Provides information about Uber quarterly financials ending June 2022",
        ),
    ),
]

# 2. Create Agent
agent = OpenAIAgent.from_tools(query_engine_tools, verbose=True)

# 3. Chat
response = agent.chat("Analyze the revenue growth from June to September.")
```

---

## 🎓 Interview Focus

1.  **What is a "Router"?**
    - A simple agent that selects *one* best tool from a list.
    - An Agent is a Router that can also loop and maintain state.

2.  **Multi-Document Agents?**
    - LlamaIndex excels here. You can have a hierarchical structure:
    - Top Agent $\to$ Sub-Agents (one per document) $\to$ Vector Search.

---

**LlamaIndex: Agents that know your data!**
