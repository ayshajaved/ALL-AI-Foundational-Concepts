# Practical Workflows: Fine-tuning ViT

> **HuggingFace for Vision** - Classifying Beans

---

## 🛠️ The Project

Fine-tune a pre-trained Vision Transformer (`google/vit-base-patch16-224`) on the **Beans** dataset (Leaf disease classification).
**Stack:** HuggingFace Transformers, Datasets, Trainer.

---

## 💻 Implementation

```python
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
import torch

# 1. Load Dataset
ds = load_dataset('beans')

# 2. Preprocess
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

def transform(example_batch):
    # Apply resizing and normalization
    inputs = processor([x for x in example_batch['image']], return_tensors='pt')
    inputs['labels'] = example_batch['labels']
    return inputs

prepared_ds = ds.with_transform(transform)

# 3. Load Model
labels = ds['train'].features['labels'].names
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)

# 4. Training Arguments
training_args = TrainingArguments(
    output_dir="./vit-beans",
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=4,
    fp16=True, # Use Mixed Precision
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False, # Important for image datasets
    push_to_hub=False,
    report_to='none',
    load_best_model_at_end=True,
)

# 5. Trainer
def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'][0] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["validation"],
)

# 6. Train
trainer.train()
```

---

## 🧠 Key Considerations

1.  **`remove_unused_columns=False`:**
    - HF Trainer tries to drop columns not in the model signature. Since our dataset has raw 'image' objects which the model doesn't accept (it wants `pixel_values`), we handle this in `collate_fn` or `transform`, but we tell Trainer not to panic.

2.  **Learning Rate:**
    - ViTs are sensitive to LR. Usually need a warmup and cosine decay.
    - Pre-trained ViTs fine-tune well with `2e-4` or `5e-5`.

3.  **Data Augmentation:**
    - ViTs overfit easily on small data. Strong augmentation (RandAugment, Mixup) is often required if training from scratch, but less critical for fine-tuning.

---

**You are now a Vision Transformer Engineer!**
