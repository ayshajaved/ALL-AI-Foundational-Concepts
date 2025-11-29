# Word Embeddings (Static)

> **Meaning in vector space** - Word2Vec, GloVe, and FastText

---

## 🎯 The Goal

Represent words as dense vectors where **semantic similarity $\approx$ geometric proximity**.
$Vector(King) - Vector(Man) + Vector(Woman) \approx Vector(Queen)$

---

## 🧠 Word2Vec (2013)

Learns embeddings by predicting context.

### 1. Skip-Gram
Predict **context** words given a **target** word.
*Input:* "fox" $\to$ *Output:* "The", "quick", "jumps"
**Better for:** Infrequent words.

### 2. CBOW (Continuous Bag of Words)
Predict **target** word given **context**.
*Input:* "The", "quick", "jumps" $\to$ *Output:* "fox"
**Better for:** Faster training.

**Implementation (Gensim):**
```python
from gensim.models import Word2Vec

sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

vector = model.wv['cat']
similar = model.wv.most_similar('cat')
```

---

## 🧤 GloVe (Global Vectors)

**Idea:** Word2Vec relies on local context windows. GloVe uses **global co-occurrence statistics**.
It factorizes the word-context co-occurrence matrix.
Often trains faster and scales better to huge corpora.

---

## ⚡ FastText (2016)

**Idea:** Represent a word as the sum of its **character n-grams**.
`apple` = `<ap`, `app`, `ppl`, `ple`, `le>`

**Superpower:** Can generate embeddings for **OOV words**!
If it sees `googleable` (never seen before), it builds the vector from `google` and `able`.

---

## 👁️ Visualizing Embeddings

Using PCA or t-SNE to project 300D vectors to 2D.

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_embeddings(model, words):
    vectors = [model.wv[w] for w in words]
    pca = PCA(n_components=2)
    result = pca.fit_transform(vectors)
    
    plt.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    plt.show()
```

---

## 🎓 Interview Focus

1.  **Word2Vec vs GloVe?**
    - Word2Vec is predictive (neural network). GloVe is count-based (matrix factorization).
    - Performance is often similar.

2.  **Why is FastText better for morphologically rich languages?**
    - Because it understands sub-word structures (prefixes/suffixes). Essential for German, Turkish, etc.

3.  **Limitation of Static Embeddings?**
    - **Polysemy:** The word "bank" has the *same* vector in "river bank" and "bank deposit".
    - Solved by Contextual Embeddings (BERT).

---

**Embeddings: Where language becomes geometry!**
