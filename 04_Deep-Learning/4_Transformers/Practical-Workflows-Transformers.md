# Practical Workflows - Transformers

> **Using HuggingFace for real-world NLP** - Fine-tuning and Inference

---

## 🛠️ HuggingFace Ecosystem

The de facto standard for Transformers.
- **Transformers:** Models and Tokenizers.
- **Datasets:** Ready-to-use datasets.
- **Accelerate:** Distributed training made easy.

---

## 📝 Project: Sentiment Analysis (Fine-tuning BERT)

### 1. Setup
```python
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset (IMDb)
dataset = load_dataset("imdb")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

### 2. Model
```python
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
```

### 3. Training (The Trainer API)
```python
training_args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
)

trainer.train()
```

### 4. Inference pipeline
```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
result = classifier("This movie was absolutely fantastic!")
print(result) # [{'label': 'LABEL_1', 'score': 0.99}]
```

---

## 👁️ Attention Visualization

Understanding what the model looks at.

```python
# Need output_attentions=True
model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
outputs = model(inputs)
attentions = outputs.attentions # List of (batch, heads, seq, seq)

# Visualize Head 0 of Layer 0
import matplotlib.pyplot as plt
import seaborn as sns

attn_map = attentions[0][0, 0, :, :].detach().numpy()
sns.heatmap(attn_map)
plt.show()
```

---

## 🚀 Optimization Tips

1.  **Mixed Precision (FP16):** Use `fp16=True` in TrainingArguments. Doubles speed, halves memory.
2.  **Gradient Accumulation:** If batch size 8 is too big for GPU, use batch size 1 and `gradient_accumulation_steps=8`.
3.  **Quantization:** Convert weights to INT8 for inference (4x smaller model).

---

**Transformers in production: From research to reality!**
