# Practical Workflows: RAG Chatbot

> **Building a "Chat with PDF" App** - LangChain & ChromaDB

---

## 🛠️ The Project

Build a chatbot that answers questions from a user-uploaded PDF.
**Stack:** LangChain, OpenAI/HuggingFace, ChromaDB.

---

## 💻 Implementation

```python
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 1. Load & Split PDF
loader = PyPDFLoader("my_document.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# 2. Embed & Store (Vector DB)
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(texts, embeddings)

# 3. Retrieval Chain
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    return_source_documents=True
)

# 4. Chat
query = "What is the main conclusion of the paper?"
result = qa_chain({"query": query})

print("Answer:", result["result"])
print("Source:", result["source_documents"][0].metadata)
```

---

## 🧩 Key Concepts

1.  **RecursiveCharacterTextSplitter:**
    - Smart splitting. Tries to split by paragraphs `\n\n`, then sentences `\n`, then spaces. Keeps semantic chunks together.
2.  **Chain Type "Stuff":**
    - Simply "stuffs" all retrieved chunks into the prompt context.
    - *Alternative:* "Map-Reduce" (summarize each chunk, then summarize summaries) for very long contexts.

---

## 🚀 Open Source Alternative (Local RAG)

Replace `ChatOpenAI` with `LlamaCpp` and `OpenAIEmbeddings` with `HuggingFaceEmbeddings`.

```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = LlamaCpp(model_path="llama-2-7b-chat.Q4_K_M.gguf")
```

---

**You just built the most popular AI app of 2024!**
