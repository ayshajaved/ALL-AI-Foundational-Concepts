# Algorithm Complexity Reference

> **Time & Space Complexity Quick Reference** - Essential for interviews and system design

---

## 📊 Machine Learning Algorithms

### Supervised Learning

| Algorithm | Training Time | Prediction Time | Space | Notes |
|-----------|--------------|-----------------|-------|-------|
| **Linear Regression** | O(n·d²) or O(n²·d) | O(d) | O(d) | Normal equation vs GD |
| **Logistic Regression** | O(n·d·i) | O(d) | O(d) | i = iterations |
| **Decision Tree** | O(n·d·log n) | O(log n) | O(n) | Balanced tree |
| **Random Forest** | O(n·d·log n·k) | O(k·log n) | O(k·n) | k = trees |
| **XGBoost/Gradient Boosting** | O(n·d·k) | O(k·log n) | O(k·n) | k = trees |
| **SVM (Linear)** | O(n·d) | O(d) | O(d) | Linear kernel |
| **SVM (RBF)** | O(n²·d) to O(n³) | O(s·d) | O(s·d) | s = support vectors |
| **KNN** | O(1) | O(n·d) | O(n·d) | Lazy learning |
| **Naive Bayes** | O(n·d) | O(c·d) | O(c·d) | c = classes |

### Unsupervised Learning

| Algorithm | Time Complexity | Space | Notes |
|-----------|----------------|-------|-------|
| **K-Means** | O(n·k·d·i) | O(n·d) | i = iterations, k = clusters |
| **Hierarchical Clustering** | O(n²·log n) | O(n²) | Agglomerative |
| **DBSCAN** | O(n·log n) | O(n) | With spatial index |
| **GMM (EM)** | O(n·k·d·i) | O(n·d) | Similar to K-Means |
| **PCA** | O(min(n²·d, n·d²)) | O(d²) | SVD-based |
| **t-SNE** | O(n²) | O(n) | Very slow for large n |
| **UMAP** | O(n^1.14) | O(n) | Faster than t-SNE |

---

## 🔥 Deep Learning Operations

### Basic Operations

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| **Matrix Multiplication** (m×n) · (n×p) | O(m·n·p) | O(m·p) | Output size |
| **Convolution** (H×W×C) * (K×K×C) | O(H·W·C·K²·F) | O(H·W·F) | F = filters |
| **Batch Normalization** | O(n·d) | O(d) | n = batch size |
| **Dropout** | O(n) | O(n) | Per neuron |
| **Softmax** | O(n) | O(n) | n = classes |
| **Attention** (seq_len = n, d_model = d) | O(n²·d) | O(n²) | Quadratic in sequence |

### Neural Network Layers

| Layer Type | Forward Pass | Backward Pass | Parameters | Notes |
|------------|-------------|---------------|------------|-------|
| **Fully Connected** | O(n·m) | O(n·m) | O(n·m) | n=input, m=output |
| **Conv2D** | O(H·W·C·K²·F) | O(H·W·C·K²·F) | O(K²·C·F) | K=kernel, F=filters |
| **MaxPool** | O(H·W·C) | O(H·W·C) | 0 | No parameters |
| **LSTM** | O(4·h²) | O(4·h²) | O(4·h²) | h = hidden size |
| **GRU** | O(3·h²) | O(3·h²) | O(3·h²) | Fewer params than LSTM |
| **Self-Attention** | O(n²·d) | O(n²·d) | O(d²) | n=seq_len, d=d_model |
| **Multi-Head Attention** | O(n²·d·h) | O(n²·d·h) | O(d²·h) | h = heads |

### Training Complexity

| Model | Training (per epoch) | Inference | Parameters |
|-------|---------------------|-----------|------------|
| **MLP** (L layers, h units) | O(L·h²·n) | O(L·h²) | O(L·h²) |
| **CNN** (L layers) | O(L·H·W·C·K²·F·n) | O(L·H·W·C·K²·F) | O(L·K²·C·F) |
| **RNN** (T timesteps) | O(T·h²·n) | O(T·h²) | O(h²) |
| **Transformer** (L layers) | O(L·n²·d·b) | O(L·n²·d) | O(L·d²) |

---

## 💬 NLP Algorithms

| Algorithm | Time | Space | Notes |
|-----------|------|-------|-------|
| **Word2Vec (Skip-gram)** | O(V·d·w·i) | O(V·d) | V=vocab, w=window, i=iters |
| **GloVe** | O(C·i) | O(V·d) | C = co-occurrences |
| **BERT Inference** | O(L·n²·d) | O(n·d) | L=layers, n=seq_len |
| **GPT Inference** | O(L·n²·d) | O(n·d) | Autoregressive |
| **Beam Search** | O(b·n·V) | O(b·n) | b=beam width, V=vocab |

---

## 👁️ Computer Vision

| Algorithm | Time | Space | Notes |
|-----------|------|-------|-------|
| **SIFT** | O(n·log n) | O(n) | n = pixels |
| **HOG** | O(n) | O(n) | Linear in pixels |
| **R-CNN** | O(r·c) | O(r) | r=regions, c=CNN cost |
| **Fast R-CNN** | O(r + c) | O(r) | Shared CNN |
| **Faster R-CNN** | O(c + n) | O(n) | RPN + detection |
| **YOLO** | O(S²·(B·5+C)) | O(S²·B) | S=grid, B=boxes, C=classes |
| **Mask R-CNN** | O(c + r·m) | O(r·m) | m=mask resolution |

---

## 🎮 Reinforcement Learning

| Algorithm | Time (per step) | Space | Notes |
|-----------|----------------|-------|-------|
| **Q-Learning** | O(1) | O(|S|·|A|) | Tabular |
| **SARSA** | O(1) | O(|S|·|A|) | Tabular |
| **DQN** | O(d) | O(|S|·d) | d = network size |
| **Policy Gradient** | O(d) | O(d) | Network forward pass |
| **A3C** | O(d·w) | O(d·w) | w = workers |
| **PPO** | O(d·b) | O(d·b) | b = batch size |

---

## 📊 Data Structures

| Structure | Access | Search | Insert | Delete | Space |
|-----------|--------|--------|--------|--------|-------|
| **Array** | O(1) | O(n) | O(n) | O(n) | O(n) |
| **Linked List** | O(n) | O(n) | O(1) | O(1) | O(n) |
| **Hash Table** | O(1)* | O(1)* | O(1)* | O(1)* | O(n) |
| **Binary Search Tree** | O(log n)* | O(log n)* | O(log n)* | O(log n)* | O(n) |
| **Heap** | O(1) | O(n) | O(log n) | O(log n) | O(n) |
| **Trie** | O(m) | O(m) | O(m) | O(m) | O(ALPHABET_SIZE·N·M) |

*Average case; worst case may differ

---

## 🔍 Search & Sort Algorithms

### Sorting

| Algorithm | Best | Average | Worst | Space | Stable |
|-----------|------|---------|-------|-------|--------|
| **Bubble Sort** | O(n) | O(n²) | O(n²) | O(1) | Yes |
| **Selection Sort** | O(n²) | O(n²) | O(n²) | O(1) | No |
| **Insertion Sort** | O(n) | O(n²) | O(n²) | O(1) | Yes |
| **Merge Sort** | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes |
| **Quick Sort** | O(n log n) | O(n log n) | O(n²) | O(log n) | No |
| **Heap Sort** | O(n log n) | O(n log n) | O(n log n) | O(1) | No |
| **Radix Sort** | O(d·n) | O(d·n) | O(d·n) | O(n+k) | Yes |

### Searching

| Algorithm | Time | Space | Notes |
|-----------|------|-------|-------|
| **Linear Search** | O(n) | O(1) | Unsorted array |
| **Binary Search** | O(log n) | O(1) | Sorted array |
| **BFS** | O(V+E) | O(V) | Graph traversal |
| **DFS** | O(V+E) | O(V) | Graph traversal |
| **A*** | O(b^d) | O(b^d) | b=branching, d=depth |
| **Dijkstra** | O((V+E) log V) | O(V) | With min-heap |

---

## 🎯 Interview Quick Reference

### Most Common Complexities

**O(1)** - Constant
- Array access
- Hash table operations (average)
- Stack/Queue operations

**O(log n)** - Logarithmic
- Binary search
- Balanced tree operations
- Heap operations

**O(n)** - Linear
- Array traversal
- Linear search
- Most single-pass algorithms

**O(n log n)** - Linearithmic
- Efficient sorting (Merge, Heap, Quick)
- Many divide-and-conquer algorithms

**O(n²)** - Quadratic
- Nested loops
- Simple sorting (Bubble, Selection, Insertion)
- Naive string matching

**O(2^n)** - Exponential
- Recursive Fibonacci (naive)
- Subset generation
- Brute force solutions

**O(n!)** - Factorial
- Permutation generation
- Traveling salesman (brute force)

---

## 💡 Optimization Tips

### When to Optimize
1. **O(n²) → O(n log n)**: Use sorting or divide-and-conquer
2. **O(n²) → O(n)**: Use hash table or two pointers
3. **O(2^n) → O(n²)**: Use dynamic programming
4. **O(n) → O(log n)**: Use binary search (if sorted)

### Space-Time Tradeoffs
- **Memoization**: Trade O(n) space for better time
- **Hash tables**: O(n) space for O(1) lookup
- **Preprocessing**: Upfront cost for faster queries

### ML-Specific Optimizations
- **Mini-batch GD**: Balance between SGD (O(d)) and Batch GD (O(n·d))
- **Approximate NN**: Use LSH or ANNOY for faster KNN
- **Model compression**: Reduce inference time/space
- **Distributed training**: Parallelize across GPUs

---

## 🎓 Interview Strategy

### When Asked About Complexity
1. **State assumptions** (input size, data structure)
2. **Analyze loops** (nested = multiply)
3. **Consider best/average/worst** cases
4. **Mention space complexity** too
5. **Suggest optimizations** if asked

### Red Flags in Interviews
- ❌ O(n²) when O(n log n) possible
- ❌ O(2^n) when DP can solve in O(n²)
- ❌ Not considering space complexity
- ❌ Not knowing standard algorithm complexities

---

**Master these complexities for technical interviews!**
