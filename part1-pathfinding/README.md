# A* Pathfinding Algorithm

This implementation provides an efficient A* pathfinding algorithm for finding optimal paths in weighted graphs using heuristic search.

## Overview

The A* algorithm combines the benefits of Dijkstra's algorithm with heuristic guidance to find optimal paths more efficiently. It uses an admissible heuristic function to estimate the cost from any node to the goal, allowing it to explore the most promising paths first.

## Features

- **Optimal Path Finding**: Guarantees shortest path when using admissible heuristics
- **Interactive Input**: Choose custom start node or use default
- **File-based I/O**: Reads graph from input file, writes results to output file
- **Romanian Cities Example**: Includes real-world map data for testing

## Algorithm Details

### Core Components
1. **Priority Queue**: Uses heapq for efficient node selection based on f-cost
2. **Heuristic Function**: Straight-line distance estimates to goal
3. **Path Reconstruction**: Maintains parent pointers for path building
4. **Visited Set**: Prevents revisiting already processed nodes

### Cost Functions
- **g(n)**: Actual cost from start to node n
- **h(n)**: Heuristic estimate from node n to goal
- **f(n) = g(n) + h(n)**: Total estimated cost through node n

## Input Format

Each line in `input.txt` represents a node:
```
NodeName HeuristicValue [Neighbor1 Cost1] [Neighbor2 Cost2] ...
```

Example:
```
Arad 366 Zerind 75 Timisoara 118 Sibiu 140
Sibiu 253 Oradea 151 Arad 140 RimnicuVilcea 80 Fagaras 99
Bucharest 0 Fagaras 211 Pitesti 101 Giurgiu 90 Urziceni 85
```

## Usage

```python
python astar_pathfinder.py
```

The program will:
1. Read the graph from `input.txt`
2. Prompt for start node (or use default)
3. Find optimal path to the goal node (heuristic value = 0)
4. Write results to `output.txt`

## Output

The algorithm produces:
- **Optimal Path**: Sequence of nodes from start to goal
- **Total Distance**: Minimum cost to reach the goal
- **Status**: Success or failure message

Example output:
```
Path: Arad -> Sibiu -> RimnicuVilcea -> Pitesti -> Bucharest
Total distance: 418
```

## Time Complexity

- **Best Case**: O(b^d) where b is branching factor, d is depth
- **Average Case**: Depends on heuristic quality
- **Space Complexity**: O(b^d) for storing nodes in memory

## Heuristic Properties

For optimal results, the heuristic should be:
- **Admissible**: Never overestimate the actual cost
- **Consistent**: h(n) ≤ cost(n,n') + h(n') for all neighbors n'

## Example Graph

The included Romanian cities example demonstrates pathfinding between major cities with realistic road distances and straight-line heuristics to Bucharest.
