"""
Comprehensive examples demonstrating all algorithms in the AI collection.
This file serves as a tutorial and testing suite for the entire repository.
"""

import sys
import os
import numpy as np

# Add parent directories to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'part1-pathfinding'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'part2-genetic-algorithm'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'part3-minimax-alpha-beta'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'part4-astar-comparison'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'part5-machine-learning'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'part6-neural-networks'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'part7-optimization-algorithms'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'part8-clustering-algorithms'))

try:
    from ml_algorithms import LinearRegression, LogisticRegression, NaiveBayes, DecisionTree
    from neural_networks import NeuralNetwork, Perceptron, create_spiral_dataset
    from optimization_algorithms import ParticleSwarmOptimization, TestFunctions
    from clustering_algorithms import KMeans, generate_cluster_data
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all algorithm files are in their respective directories.")
    sys.exit(1)


def demo_pathfinding():
    """Demonstrate A* pathfinding algorithm."""
    print("\n" + "="*60)
    print("PATHFINDING ALGORITHMS DEMO")
    print("="*60)
    
    # Create a simple graph for demonstration
    graph = {
        "A": [("B", 4), ("C", 2)],
        "B": [("D", 3), ("E", 1)],
        "C": [("D", 6), ("F", 3)],
        "D": [("G", 2)],
        "E": [("G", 4)],
        "F": [("G", 1)],
        "G": []
    }
    
    # Heuristic values (distance to goal G)
    heuristic = {"A": 6, "B": 4, "C": 4, "D": 2, "E": 4, "F": 1, "G": 0}
    
    print("Graph structure:")
    for node, edges in graph.items():
        if edges:
            edge_str = ", ".join([f"{neighbor}({cost})" for neighbor, cost in edges])
            print(f"  {node} -> {edge_str}")
        else:
            print(f"  {node} -> GOAL")
    
    print(f"\nHeuristic values: {heuristic}")
    print("\nNote: This demonstrates the concept. Full implementation in part1-pathfinding/")


def demo_genetic_algorithm():
    """Demonstrate genetic algorithm for optimization."""
    print("\n" + "="*60)
    print("GENETIC ALGORITHM DEMO")
    print("="*60)
    
    # Simple example: optimize a quadratic function
    def fitness_function(x):
        """Minimize (x - 5)^2, so optimal x = 5"""
        return -(x - 5)**2  # Negative because GA maximizes
    
    # Simple genetic algorithm implementation
    population_size = 20
    generations = 50
    mutation_rate = 0.1
    
    # Initialize population (random values between 0 and 10)
    population = [np.random.uniform(0, 10) for _ in range(population_size)]
    
    print("Simple GA optimizing f(x) = -(x-5)^2")
    print("Target: x = 5 (maximum)")
    
    for gen in range(generations):
        # Evaluate fitness
        fitness_scores = [fitness_function(individual) for individual in population]
        
        # Selection and reproduction (simplified)
        # Select top 50% and duplicate them
        sorted_indices = np.argsort(fitness_scores)[::-1]
        top_half = [population[i] for i in sorted_indices[:population_size//2]]
        
        # Create new population
        new_population = top_half * 2  # Duplicate top half
        
        # Mutation
        for i in range(len(new_population)):
            if np.random.random() < mutation_rate:
                new_population[i] += np.random.normal(0, 0.5)
                new_population[i] = np.clip(new_population[i], 0, 10)
        
        population = new_population
        
        if gen % 10 == 0:
            best_fitness = max(fitness_scores)
            best_individual = population[np.argmax(fitness_scores)]
            print(f"Generation {gen}: Best = {best_individual:.3f}, Fitness = {best_fitness:.3f}")
    
    print("\nNote: Full course scheduling GA implementation in part2-genetic-algorithm/")


def demo_minimax():
    """Demonstrate minimax algorithm with alpha-beta pruning."""
    print("\n" + "="*60)
    print("MINIMAX WITH ALPHA-BETA PRUNING DEMO")
    print("="*60)
    
    # Simple tic-tac-toe like game tree
    game_tree = [3, 12, 8, 2, 4, 6, 14, 5, 2]  # Leaf node values
    
    def minimax(depth, node_index, maximizing_player, values, alpha, beta):
        """Minimax with alpha-beta pruning."""
        # Base case: reached leaf node
        if depth == 0:
            return values[node_index]
        
        if maximizing_player:
            max_eval = float('-inf')
            for i in range(2):  # Binary tree
                eval_score = minimax(depth - 1, node_index * 2 + i, False, values, alpha, beta)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            return max_eval
        else:
            min_eval = float('inf')
            for i in range(2):
                eval_score = minimax(depth - 1, node_index * 2 + i, True, values, alpha, beta)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            return min_eval
    
    print("Game tree values:", game_tree)
    result = minimax(3, 0, True, game_tree, float('-inf'), float('inf'))
    print(f"Minimax result: {result}")
    print("\nNote: Full game implementations in part3-minimax-alpha-beta/")


def demo_machine_learning():
    """Demonstrate machine learning algorithms."""
    print("\n" + "="*60)
    print("MACHINE LEARNING ALGORITHMS DEMO")
    print("="*60)
    
    # Generate sample data
    np.random.seed(42)
    
    # Linear regression data
    X_reg = np.random.randn(100, 1)
    y_reg = 2 * X_reg.flatten() + 1 + 0.1 * np.random.randn(100)
    
    # Classification data
    X_cls = np.random.randn(100, 2)
    y_cls = (X_cls[:, 0] + X_cls[:, 1] > 0).astype(int)
    
    print("Testing Linear Regression...")
    lr = LinearRegression(learning_rate=0.1, max_iterations=1000)
    lr.fit(X_reg, y_reg)
    r2 = lr.score(X_reg, y_reg)
    print(f"R² Score: {r2:.4f}")
    
    print("\nTesting Logistic Regression...")
    log_reg = LogisticRegression(learning_rate=0.1, max_iterations=1000)
    log_reg.fit(X_cls, y_cls)
    accuracy = log_reg.accuracy(X_cls, y_cls)
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nTesting Naive Bayes...")
    nb = NaiveBayes()
    nb.fit(X_cls, y_cls)
    nb_accuracy = nb.accuracy(X_cls, y_cls)
    print(f"Accuracy: {nb_accuracy:.4f}")
    
    print("\nTesting Decision Tree...")
    dt = DecisionTree(max_depth=5)
    dt.fit(X_cls, y_cls)
    dt_accuracy = dt.accuracy(X_cls, y_cls)
    print(f"Accuracy: {dt_accuracy:.4f}")


def demo_neural_networks():
    """Demonstrate neural network implementations."""
    print("\n" + "="*60)
    print("NEURAL NETWORKS DEMO")
    print("="*60)
    
    # XOR problem - classic test for neural networks
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    
    print("Testing Neural Network on XOR problem...")
    nn = NeuralNetwork([2, 4, 1], activation='relu', learning_rate=0.1)
    nn.fit(X_xor, y_xor, epochs=2000, verbose=False)
    
    predictions = nn.predict(X_xor)
    accuracy = np.mean(predictions.flatten() == y_xor)
    print(f"XOR Accuracy: {accuracy:.4f}")
    print("XOR Predictions vs True:")
    for i in range(len(X_xor)):
        print(f"  {X_xor[i]} -> {predictions[i][0]} (true: {y_xor[i]})")
    
    print("\nTesting Perceptron on linearly separable data...")
    X_linear = np.random.randn(100, 2)
    y_linear = (X_linear[:, 0] + X_linear[:, 1] > 0).astype(int)
    
    perceptron = Perceptron(learning_rate=0.1, max_iterations=1000)
    perceptron.fit(X_linear, y_linear)
    perceptron_accuracy = perceptron.accuracy(X_linear, y_linear)
    print(f"Perceptron Accuracy: {perceptron_accuracy:.4f}")


def demo_optimization():
    """Demonstrate optimization algorithms."""
    print("\n" + "="*60)
    print("OPTIMIZATION ALGORITHMS DEMO")
    print("="*60)
    
    # Test on sphere function: f(x,y) = x² + y²
    print("Optimizing Sphere function: f(x,y) = x² + y²")
    print("Global minimum at (0, 0) with value 0")
    
    bounds = [(-5, 5), (-5, 5)]
    
    # Test PSO
    print("\nTesting Particle Swarm Optimization...")
    pso = ParticleSwarmOptimization(n_particles=20)
    best_solution, best_fitness = pso.optimize(TestFunctions.sphere, bounds, max_iterations=100)
    print(f"PSO Result: {best_solution}, Fitness: {best_fitness:.8f}")
    
    # Test with a more complex function
    print("\nTesting on Rosenbrock function...")
    print("Global minimum at (1, 1) with value 0")
    
    bounds = [(-5, 5), (-5, 5)]
    best_solution, best_fitness = pso.optimize(TestFunctions.rosenbrock, bounds, max_iterations=200)
    print(f"PSO Result: {best_solution}, Fitness: {best_fitness:.8f}")


def demo_clustering():
    """Demonstrate clustering algorithms."""
    print("\n" + "="*60)
    print("CLUSTERING ALGORITHMS DEMO")
    print("="*60)
    
    # Generate synthetic clustering data
    print("Generating synthetic data with 3 clusters...")
    X, y_true = generate_cluster_data(n_samples=150, n_centers=3, cluster_std=1.0)
    
    print("Testing K-Means clustering...")
    kmeans = KMeans(k=3, init_method='kmeans++')
    kmeans.fit(X)
    
    print(f"Found {len(set(kmeans.labels))} clusters")
    print(f"Final centroids:")
    for i, centroid in enumerate(kmeans.centroids):
        print(f"  Cluster {i}: ({centroid[0]:.2f}, {centroid[1]:.2f})")
    
    # Simple accuracy calculation (matching clusters to true labels)
    from collections import Counter
    predicted_clusters = kmeans.labels
    
    print(f"Cluster assignments distribution: {Counter(predicted_clusters)}")
    print("Note: Full clustering evaluation in part8-clustering-algorithms/")


def demo_comprehensive_workflow():
    """Demonstrate a comprehensive AI workflow using multiple algorithms."""
    print("\n" + "="*60)
    print("COMPREHENSIVE AI WORKFLOW DEMO")
    print("="*60)
    
    print("Scenario: Analyzing customer data for business insights")
    print("-" * 50)
    
    # Generate synthetic customer data
    np.random.seed(42)
    n_customers = 200
    
    # Customer features: age, income, spending_score
    age = np.random.normal(40, 12, n_customers)
    income = np.random.normal(60000, 15000, n_customers)
    spending_score = age * 0.5 + income * 0.00002 + np.random.normal(0, 10, n_customers)
    
    X = np.column_stack([age, income, spending_score])
    
    print(f"Generated data for {n_customers} customers")
    print("Features: Age, Income, Spending Score")
    
    # Step 1: Clustering to find customer segments
    print("\n1. Customer Segmentation (K-Means Clustering)")
    kmeans = KMeans(k=3, init_method='kmeans++')
    kmeans.fit(X)
    customer_segments = kmeans.labels
    
    print(f"Identified 3 customer segments:")
    for i in range(3):
        segment_customers = X[customer_segments == i]
        avg_age = np.mean(segment_customers[:, 0])
        avg_income = np.mean(segment_customers[:, 1])
        avg_spending = np.mean(segment_customers[:, 2])
        print(f"  Segment {i}: {len(segment_customers)} customers")
        print(f"    Avg Age: {avg_age:.1f}, Avg Income: ${avg_income:.0f}, Avg Spending: {avg_spending:.1f}")
    
    # Step 2: Predict high-value customers using classification
    print("\n2. High-Value Customer Prediction (Classification)")
    # Define high-value customers as those with spending score > median
    high_value = (spending_score > np.median(spending_score)).astype(int)
    
    # Use features: age and income to predict high-value status
    X_classification = X[:, :2]  # Age and income only
    
    # Split data for training and testing
    from ml_algorithms import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_classification, high_value, test_size=0.3, random_state=42
    )
    
    # Train logistic regression
    log_reg = LogisticRegression(learning_rate=0.01, max_iterations=1000)
    log_reg.fit(X_train, y_train)
    
    train_accuracy = log_reg.accuracy(X_train, y_train)
    test_accuracy = log_reg.accuracy(X_test, y_test)
    
    print(f"Logistic Regression Results:")
    print(f"  Training Accuracy: {train_accuracy:.3f}")
    print(f"  Testing Accuracy: {test_accuracy:.3f}")
    
    # Step 3: Optimize marketing budget allocation
    print("\n3. Marketing Budget Optimization")
    
    def marketing_roi(allocation):
        """Simulate ROI based on allocation to different segments."""
        # allocation[0] = segment 0, allocation[1] = segment 1, allocation[2] = segment 2
        # Assume different ROI rates for each segment
        roi_rates = [0.15, 0.22, 0.18]  # 15%, 22%, 18% ROI
        segment_sizes = [np.sum(customer_segments == i) for i in range(3)]
        
        total_roi = 0
        for i in range(3):
            segment_investment = allocation[i]
            segment_roi = roi_rates[i] * segment_investment * segment_sizes[i] / 100
            total_roi += segment_roi
        
        return -total_roi  # Negative because optimizer minimizes
    
    # Use PSO to optimize budget allocation
    budget_bounds = [(0, 100), (0, 100), (0, 100)]  # Budget per customer in each segment
    pso = ParticleSwarmOptimization(n_particles=20)
    
    print("Optimizing marketing budget allocation...")
    optimal_allocation, optimal_roi = pso.optimize(marketing_roi, budget_bounds, max_iterations=50)
    
    print(f"Optimal allocation per customer:")
    for i in range(3):
        print(f"  Segment {i}: ${optimal_allocation[i]:.2f}")
    print(f"Expected ROI: ${-optimal_roi:.2f}")
    
    print("\n4. Summary and Insights")
    print("- Customer segmentation revealed 3 distinct groups")
    print("- Classification model can predict high-value customers with good accuracy")
    print("- Optimization found best marketing budget allocation")
    print("- This workflow demonstrates integration of multiple AI techniques")


def main():
    """Main function to run all demos."""
    print("AI Algorithms Collection - Comprehensive Demo")
    print("=" * 60)
    print("This demonstration showcases the various AI algorithms implemented in this repository.")
    print("Each section demonstrates different aspects of artificial intelligence.")
    
    try:
        demo_pathfinding()
        demo_genetic_algorithm()
        demo_minimax()
        demo_machine_learning()
        demo_neural_networks()
        demo_optimization()
        demo_clustering()
        demo_comprehensive_workflow()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nExplore individual algorithm implementations in their respective directories:")
        print("- part1-pathfinding/: A* pathfinding algorithm")
        print("- part2-genetic-algorithm/: Course scheduling optimization")
        print("- part3-minimax-alpha-beta/: Game AI with minimax")
        print("- part4-astar-comparison/: A* algorithm variants")
        print("- part5-machine-learning/: Classical ML algorithms")
        print("- part6-neural-networks/: Neural network implementations")
        print("- part7-optimization-algorithms/: Advanced optimization methods")
        print("- part8-clustering-algorithms/: Unsupervised learning algorithms")
        print("\nFor datasets and additional examples, check the datasets/ and examples/ directories.")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        print("Make sure all required files are in their correct directories.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
