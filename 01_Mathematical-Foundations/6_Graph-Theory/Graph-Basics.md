# Graph Basics

> **Foundation of network analysis** - Nodes, edges, and graph representations

---

## 🎯 Graph Definitions

### Basic Concepts

**Graph:** G = (V, E)
- V: set of vertices (nodes)
- E: set of edges (connections)

**Types:**
- **Undirected:** edges have no direction
- **Directed (Digraph):** edges have direction
- **Weighted:** edges have weights
- **Unweighted:** all edges equal

---

## 📊 Graph Representations

### 1. Adjacency Matrix

```python
import numpy as np

# Undirected graph
#   0 - 1
#   |   |
#   2 - 3

A = np.array([
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0]
])

# Properties
# A[i,j] = 1 if edge (i,j) exists
# Symmetric for undirected graphs
# Space: O(V²)
```

### 2. Adjacency List

```python
# More space-efficient for sparse graphs
graph = {
    0: [1, 2],
    1: [0, 3],
    2: [0, 3],
    3: [1, 2]
}

# Space: O(V + E)
```

### 3. Edge List

```python
edges = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 3)
]

# Simple but inefficient for queries
```

---

## 🔍 Graph Traversal

### Depth-First Search (DFS)

```python
def dfs(graph, start, visited=None):
    """DFS traversal"""
    if visited is None:
        visited = set()
    
    visited.add(start)
    print(start, end=' ')
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    
    return visited

# Iterative version
def dfs_iterative(graph, start):
    """Iterative DFS using stack"""
    visited = set()
    stack = [start]
    
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            print(node, end=' ')
            stack.extend(reversed(graph[node]))
    
    return visited
```

### Breadth-First Search (BFS)

```python
from collections import deque

def bfs(graph, start):
    """BFS traversal"""
    visited = set([start])
    queue = deque([start])
    
    while queue:
        node = queue.popleft()
        print(node, end=' ')
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return visited
```

---

## 📈 Graph Properties

### Degree

```python
def degree(graph, node):
    """Degree of a node"""
    return len(graph[node])

def degree_distribution(graph):
    """Degree distribution"""
    degrees = [len(neighbors) for neighbors in graph.values()]
    return np.bincount(degrees)
```

### Connectivity

```python
def is_connected(graph):
    """Check if graph is connected"""
    if not graph:
        return True
    
    start = next(iter(graph))
    visited = dfs(graph, start)
    
    return len(visited) == len(graph)
```

---

## 🎯 Special Graphs

### Complete Graph K_n
```python
def complete_graph(n):
    """Generate complete graph"""
    return {i: [j for j in range(n) if j != i] for i in range(n)}
```

### Cycle Graph C_n
```python
def cycle_graph(n):
    """Generate cycle graph"""
    return {i: [(i-1) % n, (i+1) % n] for i in range(n)}
```

### Tree
- Connected acyclic graph
- |E| = |V| - 1

---

## 🎓 Interview Focus

### Key Questions

1. **DFS vs BFS?**
   - DFS: Stack, goes deep, O(V+E)
   - BFS: Queue, level-by-level, O(V+E)

2. **Adjacency matrix vs list?**
   - Matrix: O(V²) space, O(1) edge query
   - List: O(V+E) space, O(degree) edge query

3. **When to use graphs?**
   - Social networks
   - Knowledge graphs
   - Neural network architectures

---

## 📚 References

- **Books:** "Introduction to Algorithms" - CLRS

---

**Graphs: the foundation of network science!**
