# Text Preprocessing

> **Garbage in, garbage out** - The foundation of all NLP pipelines

---

## 🧹 The Pipeline

Raw text is messy. Before feeding it to a model, we must clean and normalize it.

### 1. Normalization
- **Lowercasing:** `Hello` $\to$ `hello` (Reduces vocabulary size).
- **Unicode Normalization:** `café` vs `cafe\u0301` (NFC normalization).
- **Removing Noise:** HTML tags, URLs, emojis (unless relevant).

### 2. Tokenization (Basic)
Splitting text into units (words, sentences).
*Note: Modern LLMs use subword tokenization (BPE), covered in the next file.*

### 3. Stopword Removal
Removing common words (`the`, `is`, `at`) that carry little meaning.
*Warning: Don't do this for Contextual Models (BERT/GPT)! They need structure.*

### 4. Stemming vs Lemmatization
- **Stemming:** Chopping off suffixes (heuristic). `running` $\to$ `run`. Fast but crude (`better` $\to$ `better`).
- **Lemmatization:** Reducing to dictionary root (morphological analysis). `better` $\to$ `good`. Accurate but slow.

---

## 💻 Implementation (Spacy & NLTK)

**Spacy** is the industry standard for production preprocessing.

```python
import spacy
import re

# Load English pipeline
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # 1. Lowercase & Remove noise
    text = text.lower()
    text = re.sub(r'<.*?>', '', text) # Remove HTML
    
    # 2. Process with Spacy
    doc = nlp(text)
    
    clean_tokens = []
    for token in doc:
        # 3. Filter Stopwords & Punctuation
        if not token.is_stop and not token.is_punct:
            # 4. Lemmatization
            clean_tokens.append(token.lemma_)
            
    return " ".join(clean_tokens)

raw_text = "<p>The running foxes are fast!</p>"
print(preprocess_text(raw_text))
# Output: "run fox fast"
```

---

## 🧩 Regular Expressions (Regex) Mastery

Essential for custom cleaning.

| Pattern | Matches | Example |
| :--- | :--- | :--- |
| `\d+` | One or more digits | `2023` |
| `\w+` | Alphanumeric word | `Python3` |
| `\s+` | Whitespace | `  ` |
| `[a-z]+` | Lowercase letters | `abc` |
| `^Start` | Start of string | |
| `End$` | End of string | |

```python
# Extract all emails
text = "Contact us at support@google.com or sales@google.com"
emails = re.findall(r'[\w\.-]+@[\w\.-]+', text)
```

---

## 🎓 Interview Focus

1.  **Stemming vs Lemmatization?**
    - Stemming is rule-based (fast, crude). Lemmatization uses a dictionary/morphology (slow, accurate).

2.  **When should you keep stopwords?**
    - When using **Contextual Models** (BERT, GPT, Llama). "To be or not to be" loses all meaning without stopwords.
    - When doing **Sentiment Analysis** (negations like "not" are crucial).

3.  **What is the problem with simple whitespace tokenization?**
    - Fails on contractions ("don't" $\to$ "don", "'t").
    - Fails on punctuation attached to words ("hello," $\to$ "hello,").

---

**Preprocessing: The unglamorous but necessary work!**
