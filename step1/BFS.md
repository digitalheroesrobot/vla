# Breadth-First Search (BFS) Algorithm

## 1. What is Breadth-First Search?

**Breadth-First Search (BFS)** is a graph traversal algorithm used to visit all the vertices (nodes) of a graph, level by level.

It starts from a selected node and visits all its neighboring nodes before moving to the next level.

**BFS uses a Queue data structure (FIFO — First In, First Out).**

## 2. Example of BFS

Consider the following graph:

```text
        A
       / \
      B   C
     / \   \
    D   E   F
```

Starting from node `A`, BFS visits the nodes in this order:

**A → B → C → D → E → F**

### Explanation

1. Visit node `A`.
2. Visit the neighbors of `A`: `B` and `C`.
3. Visit the unvisited neighbors of `B`: `D` and `E`.
4. Visit the unvisited neighbor of `C`: `F`.
5. Stop when all reachable nodes have been visited.

## 3. Steps of the BFS Algorithm

1. Start from the source node.
2. Mark the source node as visited.
3. Insert the source node into the queue.
4. While the queue is not empty:
   - Remove the front node from the queue.
   - Visit or process that node.
   - Add all its unvisited neighboring nodes to the queue.
   - Mark those neighbors as visited.
5. Stop when the queue becomes empty.

**Important:** Mark a node as visited when you enqueue it, not when you dequeue it. This prevents the same node from being added to the queue multiple times.

## 4. BFS Pseudocode

```text
BFS(Graph, start):
    Create an empty queue Q
    Create an empty visited set

    Mark start as visited
    Enqueue start into Q

    while Q is not empty:
        node = Dequeue Q
        Print node

        for each neighbor of node:
            if neighbor is not visited:
                Mark neighbor as visited
                Enqueue neighbor into Q
```

## 5. BFS Implementation in Python

```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        node = queue.popleft()
        print(node, end=" ")

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)


graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': [],
    'F': []
}

bfs(graph, 'A')
```

### Output

```text
A B C D E F
```

## 6. How the Queue Works

| Step | Node Visited | Queue After Processing |
|---|---|---|
| 1 | A | B, C |
| 2 | B | C, D, E |
| 3 | C | D, E, F |
| 4 | D | E, F |
| 5 | E | F |
| 6 | F | Empty |

The queue ensures that nodes discovered earlier are processed first.

## 7. Time and Space Complexity

For a graph represented using an adjacency list:

| Complexity | Value | Explanation |
|---|---|---|
| Time Complexity | `O(V + E)` | Each vertex and edge is processed at most a constant number of times. |
| Space Complexity | `O(V)` | The visited set and queue may store up to V vertices. |

Where:

- `V` = Number of vertices (nodes).
- `E` = Number of edges.

For a graph represented using an adjacency matrix, the time complexity is generally `O(V²)`.

## 8. Applications of BFS

BFS is commonly used for:

1. **Shortest Path:** Finding the shortest path in an unweighted graph.
2. **Web Crawling:** Exploring linked pages level by level.
3. **Social Networks:** Finding people within a certain number of connections.
4. **Network Broadcasting:** Sending information to neighboring nodes.
5. **Connected Components:** Finding reachable nodes in an undirected graph.
6. **Maze Solving:** Finding a shortest route when each move has equal cost.
7. **Tree Traversal:** Performing level-order traversal of a tree.

## 9. Difference Between BFS and DFS

| Feature | BFS | DFS |
|---|---|---|
| Full Form | Breadth-First Search | Depth-First Search |
| Traversal | Level by level | Goes deep before backtracking |
| Data Structure | Queue | Stack or recursion |
| Shortest Path | Finds shortest paths in unweighted graphs | Does not generally guarantee shortest paths |
| Memory Usage | Can be high for wide graphs | Depends on search depth and graph structure |
| Common Applications | Shortest paths, level-order traversal | Backtracking, cycle detection, topological sorting |

## 10. Advantages of BFS

- Finds the shortest path in an unweighted graph.
- Explores nodes systematically, level by level.
- Can be used to find all nodes reachable from a starting node.
- Works for both directed and undirected graphs.

## 11. Disadvantages of BFS

- May require significant memory for wide graphs.
- Can be inefficient when the target is very deep in a large graph.
- Does not directly find the shortest path in weighted graphs with arbitrary edge weights.

## 12. Key Points to Remember for Exams

- BFS stands for **Breadth-First Search**.
- It is a graph traversal algorithm.
- It uses a **Queue (FIFO)**.
- It visits nodes level by level.
- Its time complexity with an adjacency list is **`O(V + E)`**.
- Its auxiliary space complexity is **`O(V)`**.
- It finds shortest paths in unweighted graphs.
- A visited set prevents repeated processing of nodes.

---

**Conclusion:** Breadth-First Search is an important graph traversal algorithm that explores nodes level by level using a queue. It is especially useful for finding shortest paths in unweighted graphs and exploring connected nodes.