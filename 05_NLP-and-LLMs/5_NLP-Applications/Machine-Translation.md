# Machine Translation

> **Breaking language barriers** - Seq2Seq, Attention, and MarianMT

---

## 🎯 The Task

Translate a sequence from Source Language ($X$) to Target Language ($Y$).
$$ P(Y | X) = \prod_{t=1}^T P(y_t | y_{<t}, X) $$

---

## 🏗️ Evolution of NMT (Neural Machine Translation)

1.  **RNN Encoder-Decoder (2014):** Compresses sentence into a fixed vector. Fails on long sentences.
2.  **Attention (2015):** Decoder looks back at relevant source words.
3.  **Transformers (2017):** Parallel training, SOTA performance.

---

## 💻 MarianMT (HuggingFace)

MarianMT is an efficient Transformer architecture trained on the OPUS dataset (massive multilingual corpus). It is the standard for open-source translation.

```python
from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-de" # English to German

# 1. Load
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# 2. Translate
src_text = ["My name is Sarah and I live in London."]

# Prepare input (padding/truncation)
translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))

# 3. Decode
tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
print(tgt_text)
# ['Mein Name ist Sarah und ich lebe in London.']
```

---

## 🧩 Handling Long Documents

Transformers have a 512-token limit. For translating books:
1.  **Sentence Splitting:** Use NLTK/Spacy to split text into sentences.
2.  **Batching:** Translate batches of sentences.
3.  **Reconstruction:** Join translated sentences.

---

## 🎓 Interview Focus

1.  **What is the "BLEU Score"?**
    - **B**ilingual **E**valuation **U**nderstudy.
    - Measures n-gram overlap between machine output and human reference.
    - Range: 0-1 (or 0-100). >30 is understandable, >50 is good, >60 is human parity.

2.  **Why is Beam Search used in translation?**
    - Greedy decoding picks the best word at each step, but might miss the best *sentence*.
    - Beam Search keeps the top $K$ partial translations at every step, finding a better global optimum.

3.  **What is Back-Translation?**
    - A data augmentation technique. Train a Target $\to$ Source model. Translate monolingual Target data back to Source. Use this synthetic pair to train the Source $\to$ Target model.

---

**Translation: The "Hello World" of Sequence-to-Sequence!**
