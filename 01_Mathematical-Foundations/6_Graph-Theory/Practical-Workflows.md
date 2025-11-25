# Practical Workflows - Graph Theory

> **Hands-on graph analysis** - NetworkX, PyTorch Geometric, DGL

---

## 🛠️ NetworkX Basics

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create graph
G = nx.Graph()

# Add nodes
G.add_nodes_from([1, 2, 3, 4])

# Add edges
G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4)])

# Visualize
nx.draw(G, with_labels=True)
plt.show()

# Graph properties
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Degree: {dict(G.degree())}")

# Shortest path
path = nx.shortest_path(G, 1, 4)
print(f"Shortest path: {path}")

# Centrality
betweenness = nx.betweenness_centrality(G)
print(f"Betweenness: {betweenness}")
```

---

## 🔥 PyTorch Geometric

```python
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Create graph data
edge_index = torch.tensor([
    [0, 1, 1, 2],
    [1, 0, 2, 1]
], dtype=torch.long)

x = torch.tensor([
    [1, 0],
    [0, 1],
    [1, 1]
], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)

# GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GCN(2, 16, 3)
out = model(data.x, data.edge_index)
```

---

## ⚡ DGL (Deep Graph Library)

```python
import dgl
import torch

# Create graph
g = dgl.graph(([0, 1, 2], [1, 2, 3]))

# Add features
g.ndata['feat'] = torch.randn(4, 10)

# GCN layer
from dgl.nn import GraphConv

conv = GraphConv(10, 5)
h = conv(g, g.ndata['feat'])

# Message passing
import dgl.function as fn

# Define message and reduce functions
g.update_all(fn.copy_u('feat', 'm'), fn.mean('m', 'h'))
```

---

## 📊 Graph Algorithms

```python
# PageRank
pagerank = nx.pagerank(G)

# Community detection
from networkx.algorithms import community
communities = community.greedy_modularity_communities(G)

# Minimum spanning tree
mst = nx.minimum_spanning_tree(G)

# Graph coloring
coloring = nx.greedy_color(G)
```

---

## 🎯 Loading Real Datasets

```python
# PyTorch Geometric datasets
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

print(f"Nodes: {data.num_nodes}")
print(f"Edges: {data.num_edges}")
print(f"Features: {data.num_features}")
print(f"Classes: {dataset.num_classes}")

# NetworkX datasets
G = nx.karate_club_graph()
G = nx.erdos_renyi_graph(100, 0.1)
G = nx.barabasi_albert_graph(100, 3)
```

---

**Master these tools for graph-based ML!**
