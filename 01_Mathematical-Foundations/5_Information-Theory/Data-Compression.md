# Data Compression

> **Lossless and lossy compression** - Storing information efficiently

---

## 🎯 Lossless Compression

### Entropy as Lower Bound
```
Average code length ≥ H(X)

H(X): entropy of source
```

**No lossless compression can beat entropy!**

---

## 📊 Huffman Coding

### Algorithm
1. Create leaf nodes for each symbol with frequency
2. Repeatedly merge two lowest-frequency nodes
3. Assign 0/1 to left/right branches

```python
import heapq
from collections import Counter

class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq

def huffman_coding(text):
    """Build Huffman tree and generate codes"""
    # Frequency count
    freq = Counter(text)
    
    # Build heap
    heap = [HuffmanNode(char, f) for char, f in freq.items()]
    heapq.heapify(heap)
    
    # Build tree
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        
        heapq.heappush(heap, merged)
    
    # Generate codes
    root = heap[0]
    codes = {}
    
    def generate_codes(node, code=""):
        if node.char is not None:
            codes[node.char] = code
            return
        generate_codes(node.left, code + "0")
        generate_codes(node.right, code + "1")
    
    generate_codes(root)
    return codes

# Example
text = "ABRACADABRA"
codes = huffman_coding(text)
print("Huffman codes:", codes)

# Encode
encoded = "".join(codes[c] for c in text)
print(f"Encoded: {encoded}")
print(f"Original bits: {len(text) * 8}")
print(f"Compressed bits: {len(encoded)}")
```

---

## 🎯 Arithmetic Coding

### Idea
Represent entire message as interval [0, 1)

**Better than Huffman for:**
- Non-integer bit allocations
- Adaptive coding
- Small alphabets

---

## 📈 Lossy Compression

### Rate-Distortion Theory
```
R(D) = min I(X;X̂)
       subject to E[d(X,X̂)] ≤ D

R(D): minimum rate for distortion D
d(X,X̂): distortion measure
```

### Transform Coding

**JPEG, MP3 workflow:**
1. Transform (DCT, FFT)
2. Quantize coefficients
3. Entropy code

```python
from scipy.fftpack import dct, idct

def jpeg_compress(block, quality=50):
    """Simplified JPEG compression"""
    # DCT
    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
    
    # Quantization
    quant_matrix = np.ones((8, 8)) * (100 - quality)
    quantized = np.round(dct_block / quant_matrix)
    
    return quantized

def jpeg_decompress(quantized, quality=50):
    """Simplified JPEG decompression"""
    # Dequantization
    quant_matrix = np.ones((8, 8)) * (100 - quality)
    dequantized = quantized * quant_matrix
    
    # Inverse DCT
    reconstructed = idct(idct(dequantized.T, norm='ortho').T, norm='ortho')
    
    return reconstructed
```

---

## 🎓 Interview Focus

1. **Huffman vs Arithmetic?**
   - Huffman: simpler, integer bits per symbol
   - Arithmetic: optimal, fractional bits

2. **Why lossy compression?**
   - Higher compression ratios
   - Perceptual redundancy
   - Trade quality for size

---

**Compression: storing more with less!**
