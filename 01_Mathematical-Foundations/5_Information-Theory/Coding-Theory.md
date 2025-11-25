# Coding Theory

> **Error detection and correction** - Reliable communication over noisy channels

---

## 🎯 Channel Coding Basics

### Noisy Channel Model

```
Source → Encoder → Channel → Decoder → Destination
                     ↓
                   Noise
```

**Goal:** Detect and correct errors introduced by noise

---

## 📊 Hamming Distance

### Definition
Number of positions where two codewords differ

```
d(x, y) = number of positions where xᵢ ≠ yᵢ
```

**Example:**
```python
def hamming_distance(x, y):
    """Compute Hamming distance"""
    return sum(a != b for a, b in zip(x, y))

x = "1011"
y = "1001"
print(f"Hamming distance: {hamming_distance(x, y)}")  # 1
```

### Minimum Distance
```
d_min = min d(x, y) for all x ≠ y in code
```

**Detection capability:** d_min - 1 errors
**Correction capability:** ⌊(d_min - 1)/2⌋ errors

---

## 🔧 Linear Codes

### Generator Matrix
```
Encode: c = mG

m: message (k bits)
G: generator matrix (k × n)
c: codeword (n bits)
```

### Parity Check Matrix
```
Syndrome: s = Hcᵀ

H: parity check matrix
s = 0 ⟺ c is valid codeword
```

**Example: (7,4) Hamming Code**
```python
import numpy as np

# Generator matrix
G = np.array([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1]
])

# Parity check matrix
H = np.array([
    [1, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1]
])

# Encode
message = np.array([1, 0, 1, 1])
codeword = message @ G % 2
print(f"Codeword: {codeword}")

# Check
syndrome = H @ codeword % 2
print(f"Syndrome: {syndrome}")  # Should be [0, 0, 0]
```

---

## 📈 Channel Capacity

### Shannon's Theorem
```
C = max I(X;Y)

C: channel capacity (bits/channel use)
I(X;Y): mutual information
```

### Binary Symmetric Channel (BSC)
```
C = 1 - H(p)

p: crossover probability
H(p): binary entropy
```

**Example:**
```python
def binary_entropy(p):
    """Binary entropy function"""
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1-p) * np.log2(1-p)

def bsc_capacity(p):
    """BSC channel capacity"""
    return 1 - binary_entropy(p)

# Example
p = 0.1  # 10% error rate
C = bsc_capacity(p)
print(f"Capacity: {C:.3f} bits/use")  # 0.531
```

---

## 🎯 Error Correction Codes

### Repetition Code
```
Encode: 0 → 000, 1 → 111
Decode: Majority vote
```

**Rate:** 1/3 (inefficient!)

### Hamming Codes
- Single error correction
- Rate: k/(k+r+1) where 2ʳ ≥ k+r+1

### Reed-Solomon Codes
- Used in CDs, DVDs, QR codes
- Powerful burst error correction

---

## 🎓 Interview Focus

### Key Questions

1. **What is channel capacity?**
   - Maximum reliable transmission rate
   - Shannon's theorem: C = max I(X;Y)
   - Fundamental limit

2. **Hamming distance significance?**
   - Measures code strength
   - d_min determines error correction capability
   - Larger distance = better code

3. **Trade-off in coding?**
   - Rate vs error correction
   - Redundancy needed for reliability
   - Shannon limit: can approach capacity

---

## 📚 References

- **Books:** "Elements of Information Theory" - Cover & Thomas
- **Papers:** "A Mathematical Theory of Communication" - Shannon

---

**Coding theory: making communication reliable!**
