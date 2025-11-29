# Practical Workflows: Model Comparison

> **Benchmarking Architectures** - BERT vs RoBERTa vs DistilBERT on GLUE

---

## 🛠️ The Experiment

Compare 3 models on the **MRPC** (Microsoft Research Paraphrase Corpus) task.
Metric: Accuracy / F1.

---

## 💻 Implementation

```python
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import numpy as np

# 1. Setup
dataset = load_dataset("glue", "mrpc")
metric = load_metric("glue", "mrpc")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

models_to_test = ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"]
results = {}

# 2. Loop
for model_name in models_to_test:
    print(f"Training {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def preprocess(examples):
        return tokenizer(examples['sentence1'], examples['sentence2'], 
                         truncation=True, padding="max_length")
    
    encoded_dataset = dataset.map(preprocess, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    args = TrainingArguments(
        f"{model_name}-finetuned",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        fp16=True
    )
    
    trainer = Trainer(
        model, args, 
        train_dataset=encoded_dataset["train"], 
        eval_dataset=encoded_dataset["validation"],
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    eval_result = trainer.evaluate()
    results[model_name] = eval_result['eval_accuracy']

# 3. Report
print("\n--- Final Results ---")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")
```

---

## 📊 Expected Outcome

1.  **RoBERTa:** Highest accuracy (~88%).
2.  **BERT:** Baseline (~84%).
3.  **DistilBERT:** Slightly lower (~82%) but 2x faster training.

---

## 🧠 Model Selection Guide

- **Accuracy Critical?** $\to$ DeBERTa-v3 or RoBERTa-Large.
- **Latency Critical?** $\to$ DistilBERT or ONNX-quantized MiniLM.
- **Long Context?** $\to$ Longformer or BigBird.
- **Generative?** $\to$ GPT-J / Llama (Decoder).

---

**Benchmarking: Data-driven decisions!**
