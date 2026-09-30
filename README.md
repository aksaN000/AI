# AI Algorithms

Classic artificial-intelligence algorithms implemented in Python, starting from my Artificial Intelligence coursework (search, genetic algorithms, adversarial search) and extended with from-scratch machine learning, neural networks, optimisation and clustering.

## Contents

| Part | File | What it implements |
|---|---|---|
| 1 | `part1-pathfinding/astar_pathfinder.py` | A* search on a weighted graph read from `input.txt`, writing the path and cost to `output.txt` |
| 2 | `part2-genetic-algorithm/course_scheduler.py` | Genetic algorithm for course scheduling: binary chromosomes, fitness on conflicts, single- and two-point crossover, mutation |
| 3 | `part3-minimax-alpha-beta/game_ai.py` | Minimax with alpha-beta pruning, applied to two small game scenarios |
| 4 | `part4-astar-comparison/astar_comparison.py` | A* with and without a closed-set check, compared on the same graph |
| 5 | `part5-machine-learning/ml_algorithms.py` | Linear regression, logistic regression, naive Bayes and a decision tree from scratch in NumPy |
| 6 | `part6-neural-networks/neural_networks.py` | Perceptron, multilayer network with backpropagation, and an RBF network, tested on a spiral dataset |
| 7 | `part7-optimization-algorithms/optimization_algorithms.py` | Particle swarm, simulated annealing, differential evolution and ant colony optimisation on standard test functions |
| 8 | `part8-clustering-algorithms/clustering_algorithms.py` | K-means, DBSCAN, hierarchical clustering and Gaussian mixture models, with evaluation metrics |

There are also three supporting scripts:
- `examples/comprehensive_demo.py` runs a short demo of every part.
- `benchmarks/performance_benchmarks.py` times the algorithms.
- `datasets/create_datasets.py` generates small CSV datasets.

## Tech stack

Python 3 · NumPy · Matplotlib · psutil (benchmarks only)

## Running locally

```bash
git clone https://github.com/aksaN000/AI.git
cd AI
pip install numpy matplotlib psutil

# Parts 1 and 2 read input.txt from their own folder
cd part1-pathfinding && python astar_pathfinder.py && cd ..
cd part2-genetic-algorithm && python course_scheduler.py && cd ..

# Part 3 asks which player moves first
python part3-minimax-alpha-beta/game_ai.py

# Everything else
python examples/comprehensive_demo.py
```

The folders for parts 1–4 each have their own README with the input format and a walkthrough of the algorithm.

## License

MIT, see [LICENSE](LICENSE).
