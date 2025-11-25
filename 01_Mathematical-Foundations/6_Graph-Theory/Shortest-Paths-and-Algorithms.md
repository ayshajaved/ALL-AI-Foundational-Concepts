# Shortest Paths and Algorithms

> **Finding optimal paths in graphs** - Dijkstra, Bellman-Ford, and Floyd-Warshall

---

## 🎯 Dijkstra's Algorithm

### Single-Source Shortest Path (Non-negative weights)

```python
import heapq

def dijkstra(graph, start):
    """
    Dijkstra's algorithm for shortest paths
    graph: dict of dict, graph[u][v] = weight
    """
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]  # (distance, node)
    visited = set()
    
    while pq:
        curr_dist, curr_node = heapq.heappop(pq)
        
        if curr_node in visited:
            continue
        visited.add(curr_node)
        
        for neighbor, weight in graph[curr_node].items():
            distance = curr_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

# Example
graph = {
    'A': {'B': 4, 'C': 2},
    'B': {'C': 1, 'D': 5},
    'C': {'D': 8, 'E': 10},
    'D': {'E': 2},
    'E': {}
}

distances = dijkstra(graph, 'A')
print(distances)  # {'A': 0, 'B': 3, 'C': 2, 'D': 8, 'E': 10}
```

**Complexity:** O((V + E) log V) with binary heap

---

## 📊 Bellman-Ford Algorithm

### Handles Negative Weights

```python
def bellman_ford(graph, start):
    """
    Bellman-Ford algorithm
    Handles negative weights, detects negative cycles
    """
    # Initialize
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    
    # Relax edges V-1 times
    for _ in range(len(graph) - 1):
        for u in graph:
            for v, weight in graph[u].items():
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
    
    # Check for negative cycles
    for u in graph:
        for v, weight in graph[u].items():
            if distances[u] + weight < distances[v]:
                raise ValueError("Graph contains negative cycle")
    
    return distances
```

**Complexity:** O(VE)

---

## 🎯 Floyd-Warshall Algorithm

### All-Pairs Shortest Paths

```python
def floyd_warshall(graph):
    """
    Floyd-Warshall algorithm
    Returns all-pairs shortest paths
    """
    nodes = list(graph.keys())
    n = len(nodes)
    
    # Initialize distance matrix
    dist = {u: {v: float('inf') for v in nodes} for u in nodes}
    
    # Distance to self is 0
    for u in nodes:
        dist[u][u] = 0
    
    # Add edge weights
    for u in graph:
        for v, weight in graph[u].items():
            dist[u][v] = weight
    
    # Floyd-Warshall
    for k in nodes:
        for i in nodes:
            for j in nodes:
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    return dist
```

**Complexity:** O(V³)

---

## 📈 A* Search

### Heuristic-Guided Search

```python
def a_star(graph, start, goal, heuristic):
    """
    A* search algorithm
    heuristic: function that estimates distance to goal
    """
    open_set = [(0, start)]  # (f_score, node)
    came_from = {}
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph}
    f_score[start] = heuristic(start, goal)
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        
        for neighbor, weight in graph[current].items():
            tentative_g = g_score[current] + weight
            
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None  # No path found
```

---

## 🎓 Interview Focus

### Algorithm Comparison

| Algorithm | Use Case | Complexity | Negative Weights |
|-----------|----------|------------|------------------|
| Dijkstra | Single-source, non-negative | O((V+E)logV) | ❌ |
| Bellman-Ford | Single-source, any weights | O(VE) | ✅ |
| Floyd-Warshall | All-pairs | O(V³) | ✅ |
| A* | Single-pair, heuristic | O(E) best case | Depends |

### Key Questions

1. **Why Dijkstra fails with negative weights?**
   - Greedy approach assumes no better path
   - Negative weights violate this assumption

2. **When to use Floyd-Warshall?**
   - Need all-pairs distances
   - Dense graphs
   - Small graphs (V³ is expensive)

3. **A* vs Dijkstra?**
   - A* uses heuristic for guidance
   - Faster when good heuristic available
   - Dijkstra is special case (h=0)

---

**Shortest paths: navigation in the graph world!**
