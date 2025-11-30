# Memory Systems for Agents

> **Total Recall** - Short-term vs Long-term Memory

---

## 🧠 The Context Window Limit

LLMs have a fixed context window (e.g., 128k tokens).
If an agent runs for days, the conversation history will overflow.
We need **Memory Management**.

---

## 🗂️ Memory Types

1.  **Short-Term Memory (Context):**
    - The immediate conversation history.
    - **BufferMemory:** Keep last $N$ messages.
    - **SummaryMemory:** Summarize the conversation so far and inject it as a system message.

2.  **Long-Term Memory (Retrieval):**
    - Storing infinite experiences in a **Vector Database** (Pinecone, Chroma).
    - **Retrieval:** When the user asks a question, fetch the top-k most relevant past memories.

3.  **Reflection Memory (Generative Agents):**
    - Periodically pause and reflect.
    - "I have observed X, Y, Z. What does this mean?" $\to$ "Insight: The user likes Python."
    - Store the *Insight* in memory, not just the raw observations.

---

## 🏗️ The MemGPT Architecture

**MemGPT** treats the context window like **RAM** and the Vector DB like **Disk**.
The OS (Agent) decides when to:
- `core_memory_append`: Write to RAM.
- `core_memory_replace`: Edit RAM.
- `archival_memory_insert`: Write to Disk.
- `archival_memory_search`: Read from Disk.

---

## 💻 Implementation (LangChain VectorStoreRetrieverMemory)

```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 1. Setup Vector DB
vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))

# 2. Setup Memory
memory = VectorStoreRetrieverMemory(retriever=retriever)

# 3. Save Context
memory.save_context(
    {"input": "My favorite food is pizza"}, 
    {"output": "Noted."}
)

# 4. Retrieve (Simulate later time)
print(memory.load_memory_variables({"prompt": "What do I like to eat?"}))
# Output: {'history': 'User: My favorite food is pizza...'}
```

---

## 🎓 Interview Focus

1.  **Recency, Importance, Relevance?**
    - The scoring function for memory retrieval (from the *Generative Agents* paper).
    - **Score** = $\alpha \cdot \text{Recency} + \beta \cdot \text{Importance} + \gamma \cdot \text{Relevance}$.

2.  **Entity Memory?**
    - Extracting facts about specific entities (User, Company) and storing them in a Knowledge Graph, rather than just vector chunks.

---

**Memory: The difference between a chat and a relationship!**
