# HuggingFace Datasets

> **Data at scale** - Loading, streaming, and mapping terabytes of text

---

## 🎯 The Problem

Deep Learning needs massive data.
- Loading a 1TB dataset into RAM crashes the machine.
- Preprocessing text on a single CPU core is too slow.

**Solution:** Apache Arrow (Memory-mapped format).

---

## 🚀 Core Features

### 1. Loading Data
Load standard datasets from the Hub or local files.

```python
from datasets import load_dataset

# Load GLUE benchmark (MRPC task)
dataset = load_dataset("glue", "mrpc")

print(dataset)
# DatasetDict({
#     train: Dataset({ features: ['sentence1', 'sentence2', 'label', ...], num_rows: 3668 }),
#     validation: Dataset({ ... }),
#     test: Dataset({ ... })
# })
```

### 2. Map (The Powerhouse)
Apply a function to every example. Parallelized and cached.

```python
def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

# batched=True speeds up tokenization by 100x (uses Rust multithreading)
tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

### 3. Streaming (Iterating without Downloading)
Train on 1TB of data with 16GB RAM.

```python
# stream=True returns an IterableDataset
dataset = load_dataset("oscar", "unshuffled_deduplicated_en", split="train", streaming=True)

for example in dataset:
    print(example['text'])
    break
```

---

## 💾 Memory Mapping

HF Datasets uses **Memory Mapping (mmap)**.
It reads data directly from disk without loading it all into RAM.
*Zero-copy overhead.*

---

## 🎓 Interview Focus

1.  **Why use `batched=True` in `.map()`?**
    - It sends a list of examples to the tokenizer instead of one by one. HF Tokenizers (Rust) can parallelize this, resulting in massive speedups.

2.  **What is Memory Mapping?**
    - A technique where a file on disk is mapped into the application's address space. The OS handles loading pages into RAM only when needed. Allows accessing datasets larger than RAM.

3.  **Difference between `Dataset` and `IterableDataset`?**
    - `Dataset`: Random access (`data[10]`), map/filter, requires disk space.
    - `IterableDataset`: Sequential access (`next(data)`), good for streaming massive web data.

---

**Datasets: Feed the beast efficiently!**
