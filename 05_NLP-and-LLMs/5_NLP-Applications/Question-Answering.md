# Question Answering (QA)

> **Finding answers in text** - Extractive QA vs Generative QA

---

## 🎯 Types of QA

1.  **Extractive QA (SQuAD style):**
    - **Input:** Context (Paragraph) + Question.
    - **Output:** Start and End indices of the answer span in the context.
    - **Model:** BERT, RoBERTa.

2.  **Generative QA:**
    - **Input:** Context + Question.
    - **Output:** Free-form text answer.
    - **Model:** T5, GPT, RAG.

3.  **Open-Domain QA:**
    - **Input:** Question only.
    - **System:** Retriever (find docs) + Reader (extract answer).

---

## 🏗️ Extractive QA Architecture

We treat this as a **Token Classification** task.
For every token, predict:
1.  Probability of being `Start` of answer.
2.  Probability of being `End` of answer.

$$ P_{start} = \text{softmax}(H \cdot W_{start}) $$
$$ P_{end} = \text{softmax}(H \cdot W_{end}) $$

---

## 💻 HuggingFace Implementation

```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

question = "Who founded SpaceX?"
context = "SpaceX was founded in 2002 by Elon Musk with the goal of reducing space transportation costs."

inputs = tokenizer(question, context, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Get most likely start and end tokens
answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
print(tokenizer.decode(predict_answer_tokens))
# "Elon Musk"
```

---

## 🎓 Interview Focus

1.  **How does BERT handle unanswerable questions (SQuAD 2.0)?**
    - The `[CLS]` token is trained to be the "answer" if no answer exists in the context. If $P(CLS) > Threshold$, output "No Answer".

2.  **Retriever-Reader Architecture?**
    - **Retriever:** TF-IDF or Dense Passage Retrieval (DPR) finds relevant docs.
    - **Reader:** BERT extracts the answer from those docs.

3.  **Why is Generative QA becoming more popular?**
    - Extractive QA cannot answer "Yes/No" questions or questions requiring synthesis ("Compare X and Y"). Generative models can.

---

**QA: Turning documents into databases!**
