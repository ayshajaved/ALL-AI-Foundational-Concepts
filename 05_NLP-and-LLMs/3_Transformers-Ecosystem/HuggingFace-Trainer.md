# HuggingFace Trainer

> **The training loop, solved** - Callbacks, Logging, and Distributed Training

---

## 🎯 Why use Trainer?

Writing a PyTorch training loop (`for epoch... optimizer.step...`) is educational but prone to bugs:
- Forgot `model.train()`?
- Forgot `optimizer.zero_grad()`?
- How to handle Gradient Accumulation?
- How to do Mixed Precision?
- How to save checkpoints?

**The `Trainer` API handles all of this.**

---

## 💻 Implementation

```python
from transformers import Trainer, TrainingArguments

# 1. Define Hyperparameters
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=True,  # Mixed Precision
    logging_dir='./logs',
)

# 2. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics, # Custom function
)

# 3. Train
trainer.train()
```

---

## 🔄 Callbacks

Hook into the training loop to add custom behavior (Early Stopping, Logging to W&B).

```python
from transformers import EarlyStoppingCallback

trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))
```

---

## 📊 Compute Metrics

Trainer expects a function that takes predictions and labels, and returns a dict.

```python
import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```

---

## 🎓 Interview Focus

1.  **What does `fp16=True` do in TrainingArguments?**
    - Enables Automatic Mixed Precision (AMP). Uses `torch.cuda.amp` to train with 16-bit floats, saving memory and speeding up training on Tensor Cores.

2.  **How does Trainer handle distributed training?**
    - It automatically detects multiple GPUs and wraps the model in `DataParallel` or `DistributedDataParallel` (DDP) without code changes.

3.  **Gradient Accumulation in Trainer?**
    - Just set `gradient_accumulation_steps=N`. It will wait N steps before calling `optimizer.step()`.

---

**Trainer: Production-grade training in 5 lines!**
