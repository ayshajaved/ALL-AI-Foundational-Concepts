# Named Entity Recognition (NER)

> **Information Extraction** - Identifying People, Orgs, and Locations

---

## 🎯 The Task

Classify every token into a category (Person, Organization, Location, Date, etc.).
**Format:** IOB (Inside-Outside-Beginning).

Input: `Apple released the iPhone.`
Output: `B-ORG O O B-PROD`

---

## 🏗️ BERT for Token Classification

Unlike BiLSTM-CRF (which needed complex hand-crafted features), BERT learns context automatically.
We simply add a Linear Classifier on top of every token's embedding.

$$ y_i = \text{softmax}(W \cdot h_i + b) $$

---

## 💻 HuggingFace Implementation

```python
from transformers import pipeline

# Aggregation strategy: 'simple' merges subwords (Apple, ##Inc) into one entity
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

text = "Elon Musk bought Twitter in San Francisco."
entities = ner(text)

for entity in entities:
    print(f"{entity['word']}: {entity['entity_group']} ({entity['score']:.2f})")

# Elon Musk: PER (0.99)
# Twitter: ORG (0.99)
# San Francisco: LOC (0.99)
```

---

## 🧩 Challenges

1.  **Nested Entities:** "University of [New York]" - Is "New York" a Location or part of the Org? Standard NER handles flat entities only.
2.  **Ambiguity:** "Washington" (Person vs State vs City). BERT solves this via context.
3.  **Subword Tokenization:** BERT splits "HuggingFace" $\to$ "Hugging", "##Face". We only predict the label for the *first* subword ("Hugging") and ignore the rest during training.

---

## 🎓 Interview Focus

1.  **How to evaluate NER?**
    - **Span-level F1 Score.** We care if the *entire entity* "New York City" is detected correct. Getting "New York" (partial) counts as a False Negative in strict evaluation.

2.  **Why use Cased models for NER?**
    - Capitalization is a huge feature ("apple" vs "Apple"). Always use `bert-base-cased`, not `uncased`.

---

**NER: Turning unstructured text into structured data!**
