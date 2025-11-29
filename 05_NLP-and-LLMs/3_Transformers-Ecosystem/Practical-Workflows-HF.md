# Practical Workflows: Fine-Tuning BERT

> **From Pre-trained to Specialized** - A complete classification pipeline

---

## 🛠️ The Project: Fine-Tuning BERT for Spam Detection

**Goal:** Classify SMS messages as Spam/Ham.
**Model:** `distilbert-base-uncased`.

---

## 🚀 Implementation

### 1. Setup & Data
```python
from datasets import load_dataset
from transformers import AutoTokenizer

# Load Dataset
dataset = load_dataset("sms_spam", split="train").train_test_split(test_size=0.2)

# Tokenize
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess(examples):
    return tokenizer(examples["sms"], truncation=True, padding="max_length")

tokenized_data = dataset.map(preprocess, batched=True)
```

### 2. Model Initialization
```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", 
    num_labels=2,
    id2label={0: "HAM", 1: "SPAM"},
    label2id={"HAM": 0, "SPAM": 1}
)
```

### 3. Metrics
```python
import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)
```

### 4. Training
```python
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="spam_detector",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
```

### 5. Inference
```python
text = "Congratulations! You've won a $1000 Walmart gift card. Click here."

inputs = tokenizer(text, return_tensors="pt").to("cuda")
logits = model(**inputs).logits
predicted_class_id = logits.argmax().item()

print(model.config.id2label[predicted_class_id])
# Output: SPAM
```

---

## 🧠 Best Practices

1.  **Learning Rate:** BERT is sensitive. Use small LRs ($2e-5$, $3e-5$, $5e-5$).
2.  **Epochs:** Fine-tuning usually converges fast (2-4 epochs).
3.  **Freezing:** You can freeze the body and train only the head first, but full fine-tuning usually yields better results if you have enough data.

---

**You just fine-tuned a Transformer!**
