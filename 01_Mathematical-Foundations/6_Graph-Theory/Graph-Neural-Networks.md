# Graph Neural Networks

> **Deep learning on graphs** - GCNs, GraphSAGE, and attention mechanisms

---

## 🎯 Why GNNs?

**Traditional ML assumes:**
- Fixed-size inputs
- Grid structure (images) or sequences (text)

**Graphs are:**
- Variable size
- Irregular structure
- Permutation invariant

---

## 📊 Message Passing Framework

### General Form
```
h_v^(k+1) = UPDATE(h_v^(k), AGGREGATE({h_u^(k) : u ∈ N(v)}))

h_v: node embedding
N(v): neighbors of v
k: layer index
```

---

## 🎯 Graph Convolutional Networks (GCN)

### Layer Definition
```
H^(k+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(k) W^(k))

Ã = A + I (add self-loops)
D̃: degree matrix of Ã
W: learnable weights
σ: activation function
```

### Implementation
```python
import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, X, A):
        """
        X: node features (N x in_features)
        A: adjacency matrix (N x N)
        """
        # Add self-loops
        A_hat = A + torch.eye(A.size(0))
        
        # Degree matrix
        D = torch.diag(A_hat.sum(dim=1))
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0
        
        # Normalize
        A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
        
        # Apply transformation
        return torch.relu(A_norm @ self.linear(X))

class GCN(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes):
        super().__init__()
        self.gc1 = GCNLayer(in_features, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, num_classes)
    
    def forward(self, X, A):
        H = self.gc1(X, A)
        return self.gc2(H, A)
```

---

## 🚀 GraphSAGE

### Idea
Sample and aggregate from neighbors

```python
class GraphSAGELayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(2 * in_features, out_features)
    
    def forward(self, X, A, num_samples=5):
        """
        Sample neighbors and aggregate
        """
        N = X.size(0)
        H_agg = []
        
        for i in range(N):
            # Get neighbors
            neighbors = A[i].nonzero().squeeze()
            
            # Sample
            if len(neighbors) > num_samples:
                sampled = neighbors[torch.randperm(len(neighbors))[:num_samples]]
            else:
                sampled = neighbors
            
            # Aggregate (mean)
            if len(sampled) > 0:
                h_neigh = X[sampled].mean(dim=0)
            else:
                h_neigh = torch.zeros(X.size(1))
            
            # Concatenate
            h = torch.cat([X[i], h_neigh])
            H_agg.append(h)
        
        H = torch.stack(H_agg)
        return torch.relu(self.linear(H))
```

---

## 🎯 Graph Attention Networks (GAT)

### Attention Mechanism
```
α_ij = softmax(LeakyReLU(a^T [W h_i || W h_j]))

h_i' = σ(Σ_j α_ij W h_j)
```

```python
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        
        self.W = nn.Linear(in_features, num_heads * out_features)
        self.a = nn.Parameter(torch.randn(num_heads, 2 * out_features))
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(self, X, A):
        N = X.size(0)
        
        # Transform features
        H = self.W(X).view(N, self.num_heads, self.out_features)
        
        # Compute attention coefficients
        a_input = torch.cat([
            H.repeat(1, 1, N).view(N * N, self.num_heads, self.out_features),
            H.repeat(N, 1, 1)
        ], dim=2)
        
        e = self.leaky_relu((a_input * self.a).sum(dim=2))
        e = e.view(N, N, self.num_heads)
        
        # Mask non-neighbors
        attention = torch.where(A.unsqueeze(2) > 0, e, torch.tensor(-1e9))
        attention = torch.softmax(attention, dim=1)
        
        # Aggregate
        H_out = torch.matmul(attention.transpose(1, 2), H)
        
        return H_out.mean(dim=1)  # Average over heads
```

---

## 📈 Applications

### Node Classification
```python
# Train GNN for node classification
model = GCN(num_features, 64, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    
    out = model(X, A)
    loss = F.cross_entropy(out[train_mask], labels[train_mask])
    
    loss.backward()
    optimizer.step()
```

### Graph Classification
```python
# Add global pooling
class GraphClassifier(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes):
        super().__init__()
        self.gcn = GCN(in_features, hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, X, A):
        H = self.gcn(X, A)
        # Global mean pooling
        h_graph = H.mean(dim=0)
        return self.classifier(h_graph)
```

---

## 🎓 Interview Focus

### Key Questions

1. **Why GNNs?**
   - Handle irregular graph structure
   - Permutation invariant
   - Inductive learning on graphs

2. **GCN vs GraphSAGE?**
   - GCN: uses all neighbors
   - GraphSAGE: samples neighbors, scalable

3. **Attention in GAT?**
   - Learns importance of neighbors
   - Different weights for different neighbors
   - More expressive than GCN

---

## 📚 References

- **Papers:**
  - "Semi-Supervised Classification with Graph Convolutional Networks" - Kipf & Welling
  - "Inductive Representation Learning on Large Graphs" - Hamilton et al. (GraphSAGE)
  - "Graph Attention Networks" - Veličković et al.

---

**GNNs: deep learning meets graph theory!**
