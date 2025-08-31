# AI Algorithms Collection

A comprehensive collection of fundamental artificial intelligence algorithms implemented in Python. This repository contains implementations of pathfinding, optimization, and game theory algorithms commonly used in AI applications.

## Repository Structure

### Part 1: Pathfinding Algorithms
**Directory: `part1-pathfinding/`**

Implementation of the A* (A-star) pathfinding algorithm for finding optimal paths in weighted graphs.

**Features:**
- A* algorithm with heuristic-based search
- Reads graph data from input files
- Outputs optimal path and total cost
- Interactive start node selection

**Files:**
- `astar_pathfinder.py` - Main A* implementation
- `input.txt` - Sample graph data (Romanian cities example)

### Part 2: Genetic Algorithm for Course Scheduling
**Directory: `part2-genetic-algorithm/`**

Genetic algorithm implementation for solving course scheduling optimization problems.

**Features:**
- Population-based optimization
- Single-point and two-point crossover operations
- Mutation operations with configurable rates
- Fitness function for scheduling constraints
- Handles course-time slot conflicts

**Files:**
- `course_scheduler.py` - Complete genetic algorithm implementation
- `input.txt` - Course scheduling input data

### Part 3: Minimax with Alpha-Beta Pruning
**Directory: `part3-minimax-alpha-beta/`**

Game AI implementations using minimax algorithm with alpha-beta pruning optimization.

**Features:**
- Mortal Kombat tournament simulation
- Pacman decision-making with "dark magic" cost analysis
- Alpha-beta pruning for efficient tree search
- Theoretical analysis of deterministic vs stochastic environments

**Files:**
- `game_ai.py` - Minimax and alpha-beta implementations

### Part 4: A* Algorithm Comparison
**Directory: `part4-astar-comparison/`**

Comparative analysis of A* algorithm implementations with different optimization strategies.

**Features:**
- Two A* variants: with and without neighbor checking
- Performance comparison and analysis
- Detailed explanation of algorithmic differences
- Sample graph testing

**Files:**
- `astar_comparison.py` - Comparative A* implementations

## Getting Started

### Prerequisites
- Python 3.6 or higher
- No external dependencies required (uses only standard library)

### Running the Algorithms

#### Part 1 - A* Pathfinding
```bash
cd part1-pathfinding
python astar_pathfinder.py
```

#### Part 2 - Genetic Algorithm
```bash
cd part2-genetic-algorithm
python course_scheduler.py
```

#### Part 3 - Minimax Games
```bash
cd part3-minimax-alpha-beta
python game_ai.py
```

#### Part 4 - A* Comparison
```bash
cd part4-astar-comparison
python astar_comparison.py
```

## Algorithm Details

### A* Pathfinding
- Uses heuristic function to guide search
- Maintains open and closed sets for efficient exploration
- Guarantees optimal path when heuristic is admissible

### Genetic Algorithm
- Population size: 10 chromosomes
- Mutation rate: 0.1 (configurable)
- Maximum iterations: 100
- Fitness based on scheduling constraint violations

### Minimax with Alpha-Beta
- Tree depth: 3-5 levels depending on game
- Alpha-beta pruning reduces search space
- Supports both maximizing and minimizing players

## Input Formats

### A* Pathfinding Input
```
NodeName HeuristicValue [NeighborName Cost] ...
```

### Genetic Algorithm Input
```
NumberOfCourses NumberOfTimeSlots
Course1Name
Course2Name
...
```

## Technical Notes

- All algorithms use efficient data structures (heaps, sets, dictionaries)
- Memory complexity optimized for practical use cases
- Code includes comprehensive comments and documentation
- Modular design allows easy extension and modification

## Performance Characteristics

- **A* Pathfinding**: O(b^d) time complexity where b is branching factor and d is depth
- **Genetic Algorithm**: Converges typically within 50-100 generations
- **Minimax**: Exponential in tree depth, optimized with alpha-beta pruning
- **Memory Usage**: Optimized for datasets with hundreds of nodes/states

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Algorithm optimizations
- Additional test cases
- Documentation improvements
- New algorithm implementations

## Author

Implementation and documentation by the repository maintainer.
Focus on clean, educational code with practical applications.
