# AI Algorithms Collection

A comprehensive collection of artificial intelligence algorithms implemented in Python from scratch. This repository serves as both an educational resource and a practical toolkit for AI development, featuring implementations across multiple domains of artificial intelligence.

## 🌟 Repository Highlights

- **8 Algorithm Categories**: Covering the breadth of AI from pathfinding to deep learning
- **Production-Ready Code**: Clean, documented, and tested implementations
- **Educational Focus**: Detailed explanations and comprehensive documentation
- **No External Dependencies**: Pure Python implementations using only standard libraries
- **Benchmarking Suite**: Performance analysis and comparison tools
- **Real-World Examples**: Practical applications and use cases

## 📁 Repository Structure

### Part 1: Pathfinding Algorithms
**Directory: `part1-pathfinding/`**

Advanced pathfinding using the A* algorithm with optimizations for real-world applications.

**Features:**
- A* algorithm with heuristic-based search
- Interactive graph exploration
- Romanian cities real-world dataset
- Performance optimization techniques

**Files:**
- `astar_pathfinder.py` - Optimized A* implementation
- `input.txt` - Real-world graph data
- `README.md` - Detailed algorithm analysis

### Part 2: Genetic Algorithm Optimization
**Directory: `part2-genetic-algorithm/`**

Evolutionary computation for complex optimization problems including course scheduling.

**Features:**
- Multi-objective optimization
- Advanced crossover strategies (single-point, two-point)
- Adaptive mutation rates
- Constraint satisfaction handling
- Population diversity maintenance

**Files:**
- `course_scheduler.py` - Complete genetic algorithm framework
- `input.txt` - Optimization problem instances
- `README.md` - Evolutionary algorithms guide

### Part 3: Game AI with Minimax
**Directory: `part3-minimax-alpha-beta/`**

Strategic game AI using minimax algorithm with advanced pruning techniques.

**Features:**
- Alpha-beta pruning optimization
- Multi-game implementations (Mortal Kombat, Pacman)
- Theoretical analysis of game trees
- Stochastic vs deterministic environments
- Performance optimization strategies

**Files:**
- `game_ai.py` - Advanced game AI implementations
- `README.md` - Game theory and implementation guide

### Part 4: A* Algorithm Analysis
**Directory: `part4-astar-comparison/`**

Comprehensive comparison and analysis of A* algorithm variants and optimizations.

**Features:**
- Multiple A* implementations
- Performance benchmarking
- Algorithmic complexity analysis
- Optimization strategy comparison

**Files:**
- `astar_comparison.py` - Algorithm comparison framework
- `README.md` - Comparative analysis documentation

### Part 5: Machine Learning Algorithms 🆕
**Directory: `part5-machine-learning/`**

Classical machine learning algorithms implemented from scratch with modern optimizations.

**Features:**
- Linear and Logistic Regression with gradient descent
- Naive Bayes classifier with Gaussian assumptions
- Decision Trees with CART algorithm
- Cross-validation and model evaluation
- Feature normalization and preprocessing

**Algorithms:**
- Linear Regression, Logistic Regression
- Naive Bayes, Decision Trees
- Model evaluation metrics
- Data preprocessing utilities

**Files:**
- `ml_algorithms.py` - Complete ML algorithm suite
- `README.md` - Machine learning theory and practice

### Part 6: Neural Networks 🆕
**Directory: `part6-neural-networks/`**

Neural network implementations from perceptrons to deep multi-layer networks.

**Features:**
- Multi-layer perceptron with customizable architectures
- Multiple activation functions (ReLU, Sigmoid, Tanh)
- Backpropagation with gradient descent
- RBF networks for non-linear problems
- XOR and spiral dataset solutions

**Algorithms:**
- Multi-layer Neural Networks
- Single-layer Perceptron
- Radial Basis Function (RBF) Networks
- Custom activation functions

**Files:**
- `neural_networks.py` - Complete neural network framework
- `README.md` - Neural networks theory and implementation

### Part 7: Optimization Algorithms 🆕
**Directory: `part7-optimization-algorithms/`**

Advanced optimization algorithms for complex problem solving.

**Features:**
- Particle Swarm Optimization (PSO)
- Simulated Annealing with adaptive cooling
- Differential Evolution strategies
- Ant Colony Optimization for continuous problems
- Benchmark test functions suite

**Algorithms:**
- Particle Swarm Optimization
- Simulated Annealing
- Differential Evolution
- Ant Colony Optimization

**Files:**
- `optimization_algorithms.py` - Advanced optimization suite
- `README.md` - Optimization theory and applications

### Part 8: Clustering Algorithms 🆕
**Directory: `part8-clustering-algorithms/`**

Unsupervised learning algorithms for pattern discovery and data analysis.

**Features:**
- K-Means with K-Means++ initialization
- DBSCAN for density-based clustering
- Hierarchical clustering with multiple linkage methods
- Gaussian Mixture Models with EM algorithm
- Comprehensive evaluation metrics

**Algorithms:**
- K-Means Clustering
- DBSCAN (Density-Based)
- Hierarchical Clustering
- Gaussian Mixture Models

**Files:**
- `clustering_algorithms.py` - Complete clustering framework
- `README.md` - Unsupervised learning guide

##  Additional Resources

### Datasets
**Directory: `datasets/`**
- Sample datasets for testing and validation
- Real-world data examples
- Synthetic data generation utilities

### Examples  
**Directory: `examples/`**
- Comprehensive demonstration scripts
- Integration examples
- Tutorial workflows

### Benchmarks
**Directory: `benchmarks/`**
- Performance analysis tools
- Algorithm comparison frameworks
- Scalability testing suites

## Getting Started

### Prerequisites
- Python 3.6 or higher
- No external dependencies required (uses only standard library)
- Optional: matplotlib for visualization (benchmarking)

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/aksaN000/AI.git
cd AI
```

2. **Run algorithm demonstrations**
```bash
# Comprehensive demo of all algorithms
python examples/comprehensive_demo.py

# Individual algorithm examples
python part1-pathfinding/astar_pathfinder.py
python part5-machine-learning/ml_algorithms.py
python part6-neural-networks/neural_networks.py
```

3. **Generate datasets**
```bash
cd datasets
python create_datasets.py
```

4. **Run benchmarks**
```bash
cd benchmarks
python performance_benchmarks.py
```

### Algorithm Usage Examples

#### Machine Learning
```python
from part5-machine-learning.ml_algorithms import LinearRegression, LogisticRegression

# Linear Regression
lr = LinearRegression(learning_rate=0.01, max_iterations=1000)
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)

# Logistic Regression
log_reg = LogisticRegression(learning_rate=0.01)
log_reg.fit(X_train, y_train)
accuracy = log_reg.accuracy(X_test, y_test)
```

#### Neural Networks
```python
from part6-neural-networks.neural_networks import NeuralNetwork

# Create and train neural network
nn = NeuralNetwork([2, 4, 1], activation='relu')
nn.fit(X_train, y_train, epochs=1000)
predictions = nn.predict(X_test)
```

#### Optimization
```python
from part7-optimization-algorithms.optimization_algorithms import ParticleSwarmOptimization

# Optimize a function
pso = ParticleSwarmOptimization(n_particles=30)
best_solution, best_fitness = pso.optimize(objective_function, bounds, max_iterations=500)
```

#### Clustering
```python
from part8-clustering-algorithms.clustering_algorithms import KMeans

# Perform clustering
kmeans = KMeans(k=3, init_method='kmeans++')
kmeans.fit(X)
cluster_labels = kmeans.labels
```

##  Performance Characteristics

### Scalability Analysis

| Algorithm | Time Complexity | Space Complexity | Recommended Dataset Size |
|-----------|----------------|------------------|--------------------------|
| A* Pathfinding | O(b^d) | O(b^d) | < 10,000 nodes |
| Genetic Algorithm | O(g × p × n) | O(p × n) | < 1,000 variables |
| Neural Networks | O(e × n × w) | O(w) | < 100,000 samples |
| K-Means | O(i × k × n) | O(k × n) | < 1,000,000 points |
| PSO | O(i × p × d) | O(p × d) | < 100 dimensions |

*Where: b=branching factor, d=depth, g=generations, p=population, n=features, e=epochs, w=weights, i=iterations, k=clusters*

### Benchmark Results Summary

- **Fastest Algorithms**: Linear models, K-Means
- **Most Accurate**: Neural Networks, Decision Trees
- **Most Memory Efficient**: Linear Regression, Perceptron
- **Best for Large Data**: K-Means, Stochastic algorithms

## Applications and Use Cases

### Real-World Applications

**Pathfinding and Navigation**
- GPS navigation systems
- Video game AI pathfinding
- Robotics motion planning
- Network routing protocols

**Machine Learning Applications**
- Customer segmentation and analysis
- Predictive modeling and forecasting
- Pattern recognition and classification
- Recommendation systems

**Optimization Solutions**
- Supply chain optimization
- Portfolio management
- Engineering design problems
- Resource allocation

**Neural Network Applications**
- Image recognition and computer vision
- Natural language processing
- Speech recognition
- Game AI and strategy

### Educational Applications

**Computer Science Curriculum**
- Algorithm design and analysis
- Artificial intelligence fundamentals
- Machine learning concepts
- Optimization theory

**Research Applications**
- Algorithm performance studies
- Comparative analysis research
- Hybrid algorithm development
- Benchmarking new approaches

## 🔬 Technical Deep Dive

### Implementation Philosophy

1. **Educational First**: Code prioritizes clarity and understanding
2. **No Black Boxes**: All algorithms implemented from fundamental principles
3. **Comprehensive Documentation**: Theory, implementation, and usage guides
4. **Practical Examples**: Real-world datasets and problem scenarios
5. **Performance Aware**: Optimized implementations with complexity analysis

### Code Quality Standards

- **PEP 8 Compliant**: Consistent Python coding standards
- **Type Hints**: Enhanced code readability and IDE support
- **Comprehensive Comments**: Detailed explanations of complex algorithms
- **Modular Design**: Easy to extend and modify implementations
- **Error Handling**: Robust error handling and edge case management

### Testing and Validation

- **Unit Testing**: Individual algorithm component validation
- **Integration Testing**: Cross-algorithm workflow testing
- **Performance Testing**: Scalability and efficiency analysis
- **Accuracy Validation**: Comparison with reference implementations

## 📈 Advanced Features

### Optimization Techniques

- **Early Stopping**: Prevent overfitting in iterative algorithms
- **Adaptive Parameters**: Dynamic learning rates and population sizes
- **Initialization Strategies**: Smart parameter initialization (Xavier, K-Means++)
- **Regularization**: L1/L2 regularization for machine learning models

### Visualization and Analysis

- **Training Curves**: Loss and accuracy visualization over time
- **Decision Boundaries**: Classifier decision region plotting
- **Cluster Visualization**: 2D/3D cluster analysis plots
- **Performance Metrics**: Comprehensive evaluation dashboards

### Extensibility Framework

- **Plugin Architecture**: Easy addition of new algorithms
- **Callback System**: Custom hooks for training monitoring
- **Configuration Management**: YAML/JSON configuration support
- **Export Capabilities**: Model serialization and deployment

## 🤝 Contributing to the AI Community

### Open Source Contributions

This repository is designed to contribute to the broader AI community by:

1. **Educational Resource**: Helping students and practitioners understand AI algorithms
2. **Research Platform**: Providing baseline implementations for research
3. **Development Tool**: Offering production-ready algorithm implementations
4. **Benchmarking Standard**: Establishing performance comparison baselines

### Community Impact

- **Algorithm Accessibility**: Making AI algorithms accessible to all skill levels
- **Best Practices**: Demonstrating clean, efficient algorithm implementations
- **Knowledge Sharing**: Comprehensive documentation and tutorials
- **Innovation Platform**: Foundation for developing hybrid and novel algorithms

## 📚 Learning Resources

### Recommended Reading

**Foundational Texts**
- "Artificial Intelligence: A Modern Approach" by Russell & Norvig
- "Pattern Recognition and Machine Learning" by Christopher Bishop
- "The Elements of Statistical Learning" by Hastie, Tibshirani & Friedman

**Algorithm-Specific Resources**
- "Introduction to Algorithms" by Cormen, Leiserson, Rivest & Stein
- "Machine Learning Yearning" by Andrew Ng
- "Deep Learning" by Ian Goodfellow, Yoshua Bengio & Aaron Courville

**Online Learning**
- Stanford CS229 Machine Learning Course
- MIT 6.034 Artificial Intelligence
- Coursera Machine Learning Specializations

### Paper References

1. **A* Algorithm**: Hart, P.E., Nilsson, N.J. and Raphael, B. (1968)
2. **Genetic Algorithms**: Holland, J.H. (1992) "Adaptation in Natural and Artificial Systems"
3. **Minimax**: Shannon, C.E. (1950) "Programming a Computer for Playing Chess"
4. **Neural Networks**: Rumelhart, D.E. et al. (1986) "Learning representations by back-propagating errors"
5. **K-Means**: MacQueen, J. (1967) "Some methods for classification and analysis"

## 🛡️ License and Citation

### License Information

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MIT License allows:
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use

### Citation

If you use this repository in your research or educational work, please cite:

```bibtex
@misc{ai_algorithms_collection,
  title={AI Algorithms Collection: Comprehensive Implementation of Fundamental AI Algorithms},
  author={Aksan Gony Alif},
  year={2024},
  url={https://github.com/aksaN000/AI},
  note={Open-source implementation of pathfinding, machine learning, neural networks, optimization, and clustering algorithms}
}
```

### Acknowledgments

- Inspired by classical AI algorithms and modern machine learning techniques
- Built for educational purposes and community contribution
- Special thanks to the open-source AI community for continuous innovation

## 📧 Contact and Support

### Getting Help

1. **Issues and Bugs**: Please use the [GitHub Issues](https://github.com/aksaN000/AI/issues) page
2. **Feature Requests**: Submit enhancement requests through GitHub Issues
3. **Documentation**: Check the individual README files in each algorithm directory
4. **Examples**: Refer to the `examples/` directory for usage demonstrations

### Contributing Guidelines

We welcome contributions! Please see our contribution guidelines:

1. **Fork the Repository**: Create your own fork for development
2. **Create Feature Branch**: Use descriptive branch names
3. **Follow Code Standards**: Maintain PEP 8 compliance and add comments
4. **Add Tests**: Include unit tests for new algorithms
5. **Update Documentation**: Keep README files current
6. **Submit Pull Request**: Provide clear description of changes

### Community

- **GitHub Discussions**: Share ideas and ask questions
- **Educational Use**: Feel free to use in courses and workshops
- **Research Applications**: Suitable for academic research and comparison studies

---

##  Final Notes

This AI Algorithms Collection represents a comprehensive journey through fundamental artificial intelligence and machine learning concepts. Each implementation prioritizes:

- **Educational Value**: Clear, well-documented code that teaches algorithmic thinking
- **Practical Application**: Real-world examples and use cases
- **Community Contribution**: Open-source availability for global learning
- **Innovation Foundation**: Base implementations for developing new algorithms

Whether you're a student learning AI concepts, a researcher developing new algorithms, or a practitioner implementing solutions, this collection provides a solid foundation for your AI journey.

**Happy coding and AI exploring!** 

---

*Last Updated: 2025 | Repository: [https://github.com/aksaN000/AI](https://github.com/aksaN000/AI) | License: MIT*
