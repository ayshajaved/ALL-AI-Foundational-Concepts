# HuggingFace Transformers

> **The standard library of NLP** - AutoClasses, Pipelines, and Tokenizers

---

## 🚀 The Ecosystem

HuggingFace (HF) has democratized NLP. It provides:
1.  **Model Hub:** 500k+ pre-trained models (BERT, GPT, Llama, Whisper).
2.  **Transformers Library:** Unified API for PyTorch/TensorFlow/JAX.
3.  **Tokenizers:** Fast Rust-based tokenization.

---

## 🏗️ Core Components

### 1. The Pipeline API (Zero-Shot)
The easiest way to use models. Handles preprocessing, inference, and postprocessing.

```python
from transformers import pipeline

# Sentiment Analysis
classifier = pipeline("sentiment-analysis")
print(classifier("I love using Transformers!"))
# [{'label': 'POSITIVE', 'score': 0.99}]

# Zero-Shot Classification (Magic!)
classifier = pipeline("zero-shot-classification")
print(classifier(
    "This is a course about Python programming",
    candidate_labels=["education", "politics", "business"]
))
# {'labels': ['education', ...], 'scores': [0.99, ...]}
```

### 2. AutoClasses (Under the Hood)
`AutoModel` and `AutoTokenizer` automatically load the correct architecture based on the checkpoint name.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Load Model
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 3. Inference
inputs = tokenizer("I am happy", return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
```

---

## 🧩 Key Architectures

- **`AutoModel`**: Base Transformer (outputs hidden states).
- **`AutoModelForSequenceClassification`**: Base + Linear Head (for classification).
- **`AutoModelForCausalLM`**: Base + Language Modeling Head (for GPT/Llama generation).
- **`AutoModelForTokenClassification`**: Base + Token Head (for NER).

---

## 💾 Saving and Loading

```python
# Save locally
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")

# Load from local
model = AutoModel.from_pretrained("./my_model")
```

---

## 🎓 Interview Focus

1.  **What is the difference between `AutoModel` and `AutoModelForX`?**
    - `AutoModel` returns the raw hidden states (embeddings).
    - `AutoModelForX` adds a specific "head" (linear layer) on top for a task (classification, generation, etc.).

2.  **What does `from_pretrained` do?**
    - Downloads the model configuration (`config.json`) and weights (`pytorch_model.bin`) from the Hub and caches them.

3.  **Why use the Pipeline API?**
    - Rapid prototyping. It abstracts away tokenization, device placement, and batching.

---

**HuggingFace: The GitHub of AI models!**
