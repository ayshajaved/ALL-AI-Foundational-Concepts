# BERT and GPT

> **The Era of Large Language Models** - Encoder vs Decoder architectures

---

## 🤖 BERT (Bidirectional Encoder Representations from Transformers)

**Architecture:** Transformer **Encoder** stack.
**Goal:** Understanding (NLU).
**Context:** Bidirectional (sees both left and right context).

### Pre-training Tasks
1.  **Masked Language Modeling (MLM):** Randomly mask 15% of tokens and predict them.
    - Input: `The [MASK] sat on the mat.`
    - Target: `cat`
2.  **Next Sentence Prediction (NSP):** Do sentences A and B follow each other?

**Use Cases:** Classification, NER, QA (Fine-tuning).

```python
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

text = "Hello, my dog is cute"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# Last hidden state: (batch, seq_len, hidden_size)
print(outputs.last_hidden_state.shape) 
```

---

## 🦄 GPT (Generative Pre-trained Transformer)

**Architecture:** Transformer **Decoder** stack.
**Goal:** Generation (NLG).
**Context:** Unidirectional (Autoregressive - sees only left context).

### Pre-training Task
1.  **Causal Language Modeling (CLM):** Predict the next token given previous tokens.
    - Input: `The cat sat on`
    - Target: `the`

**Use Cases:** Text generation, completion, zero-shot tasks.

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

text = "The AI revolution is"
input_ids = tokenizer.encode(text, return_tensors='pt')

# Generate
output = model.generate(input_ids, max_length=20)
print(tokenizer.decode(output[0]))
```

---

## ⚔️ BERT vs GPT Comparison

| Feature | BERT | GPT |
| :--- | :--- | :--- |
| **Part** | Encoder | Decoder |
| **Direction** | Bidirectional | Unidirectional (Left-to-Right) |
| **Objective** | MLM (Fill in the blank) | CLM (Next word prediction) |
| **Strength** | Understanding, Classification | Generation, Creativity |
| **Example** | "The [MASK] is blue" | "The sky is..." |

---

## 🔄 T5 (Text-to-Text Transfer Transformer)

**Architecture:** Encoder-Decoder (Full Transformer).
**Idea:** Treat *every* NLP task as a text-to-text problem.
- Translation: "translate English to German: ..."
- Classification: "cola sentence: ..." -> "acceptable"

---

## 🎓 Interview Focus

1.  **Why can't GPT be bidirectional?**
    - If GPT could see the future tokens (right context), the pre-training task (predict next word) would be trivial (cheating).

2.  **Why is BERT better for classification than GPT?**
    - BERT builds a representation of the *entire* sentence at once, capturing full context. GPT only captures context up to the current token.

3.  **What is the [CLS] token in BERT?**
    - A special token added to the start of every sequence. Its final hidden state is used as the aggregate representation of the sequence for classification tasks.

---

**BERT & GPT: The two kings of NLP!**
