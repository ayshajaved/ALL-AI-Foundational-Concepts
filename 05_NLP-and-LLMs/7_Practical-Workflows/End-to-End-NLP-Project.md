# End-to-End NLP Project

> **From Notebook to Production** - Serving a Model with FastAPI and Docker

---

## 🎯 The Goal

Deploy a Sentiment Analysis model as a REST API.
**Stack:** PyTorch, HuggingFace, FastAPI, Docker.

---

## 1. The Application (`main.py`)

FastAPI is the standard for Python AI microservices (async, fast, auto-docs).

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Sentiment API")

# Load model at startup (Global variable)
# In production, use a proper lifespan handler
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

class Request(BaseModel):
    text: str

class Response(BaseModel):
    label: str
    score: float

@app.post("/predict", response_model=Response)
async def predict(request: Request):
    if not request.text:
        raise HTTPException(status_code=400, detail="Empty text")
    
    result = classifier(request.text)[0]
    return Response(label=result['label'], score=result['score'])

@app.get("/health")
def health():
    return {"status": "ok"}
```

---

## 2. The Container (`Dockerfile`)

```dockerfile
# Use lightweight Python image
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
# torch cpu version to save space (if no GPU)
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY main.py .

# Expose port
EXPOSE 8000

# Run with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`requirements.txt`:
```text
fastapi
uvicorn
transformers
torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 3. Deployment

**Build:**
```bash
docker build -t sentiment-api .
```

**Run:**
```bash
docker run -p 8000:8000 sentiment-api
```

**Test:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "I love deploying AI models!"}'
```

---

## 🎓 Interview Focus

1.  **Why FastAPI over Flask?**
    - **Async:** Handles concurrent requests better (crucial for I/O bound tasks).
    - **Pydantic:** Automatic data validation.
    - **Swagger UI:** Auto-generated documentation at `/docs`.

2.  **How to handle heavy models?**
    - Don't load the model inside the request function (latency). Load it once at startup.
    - For high traffic, use a dedicated serving engine like **TorchServe** or **Triton**, or **vLLM** for LLMs.

---

**You have moved from "It works on my machine" to "It works everywhere"!**
