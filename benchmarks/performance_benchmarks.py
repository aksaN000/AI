"""
Benchmarking suite for AI algorithms collection.
Provides performance analysis and comparison tools for all implemented algorithms.
"""

import time
import psutil
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable, Any
import sys
import json


class AlgorithmBenchmark:
    """
    Benchmark class for measuring algorithm performance.
    Tracks execution time, memory usage, and accuracy metrics.
    """
    
    def __init__(self, algorithm_name: str):
        self.algorithm_name = algorithm_name
        self.results = {
            'execution_times': [],
            'memory_usage': [],
            'accuracy_scores': [],
            'dataset_sizes': [],
            'additional_metrics': {}
        }
    
    def benchmark_algorithm(self, algorithm_func: Callable, datasets: List[Tuple], 
                          metric_func: Callable = None, iterations: int = 5) -> Dict[str, Any]:
        """
        Benchmark an algorithm across multiple datasets and iterations.
        
        Args:
            algorithm_func: Function that implements the algorithm
            datasets: List of (X, y) tuples for testing
            metric_func: Function to compute accuracy/performance metric
            iterations: Number of iterations per dataset size
            
        Returns:
            Dictionary with benchmark results
        """
        print(f"Benchmarking {self.algorithm_name}...")
        
        for dataset_idx, (X, y) in enumerate(datasets):
            dataset_size = len(X)
            print(f"  Testing on dataset {dataset_idx + 1} (size: {dataset_size})")
            
            iteration_times = []
            iteration_memory = []
            iteration_accuracy = []
            
            for iteration in range(iterations):
                # Measure memory before execution
                process = psutil.Process(os.getpid())
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                
                # Time the algorithm execution
                start_time = time.time()
                
                try:
                    result = algorithm_func(X, y)
                    execution_time = time.time() - start_time
                    
                    # Measure memory after execution
                    memory_after = process.memory_info().rss / 1024 / 1024  # MB
                    memory_used = memory_after - memory_before
                    
                    # Calculate accuracy if metric function provided
                    accuracy = None
                    if metric_func and result is not None:
                        accuracy = metric_func(result, X, y)
                    
                    iteration_times.append(execution_time)
                    iteration_memory.append(max(0, memory_used))  # Ensure non-negative
                    if accuracy is not None:
                        iteration_accuracy.append(accuracy)
                    
                except Exception as e:
                    print(f"    Error in iteration {iteration + 1}: {e}")
                    continue
            
            # Store average results for this dataset size
            if iteration_times:
                self.results['execution_times'].append(np.mean(iteration_times))
                self.results['memory_usage'].append(np.mean(iteration_memory))
                self.results['dataset_sizes'].append(dataset_size)
                
                if iteration_accuracy:
                    self.results['accuracy_scores'].append(np.mean(iteration_accuracy))
        
        return self.results
    
    def save_results(self, filename: str) -> None:
        """Save benchmark results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def load_results(self, filename: str) -> None:
        """Load benchmark results from JSON file."""
        with open(filename, 'r') as f:
            self.results = json.load(f)


class MachineLearningBenchmarks:
    """Benchmarking suite specifically for machine learning algorithms."""
    
    @staticmethod
    def generate_classification_datasets(sizes: List[int] = [100, 500, 1000, 2000]) -> List[Tuple]:
        """Generate classification datasets of varying sizes."""
        datasets = []
        np.random.seed(42)
        
        for size in sizes:
            # Generate linearly separable data
            X = np.random.randn(size, 2)
            y = (X[:, 0] + X[:, 1] + np.random.normal(0, 0.1, size) > 0).astype(int)
            datasets.append((X, y))
        
        return datasets
    
    @staticmethod
    def generate_regression_datasets(sizes: List[int] = [100, 500, 1000, 2000]) -> List[Tuple]:
        """Generate regression datasets of varying sizes."""
        datasets = []
        np.random.seed(42)
        
        for size in sizes:
            X = np.random.randn(size, 3)
            y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.normal(0, 0.1, size)
            datasets.append((X, y))
        
        return datasets
    
    @staticmethod
    def classification_accuracy(model, X, y):
        """Calculate classification accuracy."""
        try:
            if hasattr(model, 'predict'):
                predictions = model.predict(X)
                return np.mean(predictions == y)
            return None
        except:
            return None
    
    @staticmethod
    def regression_r2_score(model, X, y):
        """Calculate R² score for regression."""
        try:
            if hasattr(model, 'score'):
                return model.score(X, y)
            elif hasattr(model, 'predict'):
                predictions = model.predict(X)
                ss_res = np.sum((y - predictions) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                return 1 - (ss_res / ss_tot)
            return None
        except:
            return None


class OptimizationBenchmarks:
    """Benchmarking suite for optimization algorithms."""
    
    @staticmethod
    def generate_optimization_problems(dimensions: List[int] = [2, 5, 10, 20]) -> List[Tuple]:
        """Generate optimization problems of varying dimensionality."""
        problems = []
        
        for dim in dimensions:
            # Sphere function with bounds
            bounds = [(-5.12, 5.12)] * dim
            problems.append((TestFunctions.sphere, bounds, dim))
        
        return problems
    
    @staticmethod
    def optimization_quality(result, objective_func, bounds):
        """Measure optimization quality (closeness to global optimum)."""
        try:
            best_solution, best_fitness = result
            # For sphere function, global minimum is 0 at origin
            return -best_fitness  # Convert to maximization problem for consistency
        except:
            return None


class ClusteringBenchmarks:
    """Benchmarking suite for clustering algorithms."""
    
    @staticmethod
    def generate_clustering_datasets(sizes: List[int] = [100, 500, 1000, 2000], 
                                   n_clusters: int = 3) -> List[Tuple]:
        """Generate clustering datasets of varying sizes."""
        datasets = []
        np.random.seed(42)
        
        for size in sizes:
            # Generate data with known clusters
            X = []
            y = []
            samples_per_cluster = size // n_clusters
            
            for cluster_id in range(n_clusters):
                # Random cluster center
                center = np.random.uniform(-10, 10, 2)
                # Generate points around center
                cluster_points = np.random.normal(center, 1.5, (samples_per_cluster, 2))
                X.extend(cluster_points)
                y.extend([cluster_id] * samples_per_cluster)
            
            # Add remaining points to last cluster
            remaining = size - len(X)
            if remaining > 0:
                center = np.random.uniform(-10, 10, 2)
                additional_points = np.random.normal(center, 1.5, (remaining, 2))
                X.extend(additional_points)
                y.extend([n_clusters - 1] * remaining)
            
            X = np.array(X)
            y = np.array(y)
            
            # Shuffle data
            indices = np.random.permutation(len(X))
            datasets.append((X[indices], y[indices]))
        
        return datasets
    
    @staticmethod
    def clustering_accuracy(model, X, y):
        """Calculate clustering accuracy (simplified)."""
        try:
            if hasattr(model, 'labels_'):
                predictions = model.labels_
            elif hasattr(model, 'predict'):
                predictions = model.predict(X)
            else:
                return None
            
            # Simple accuracy calculation (best label mapping)
            from itertools import permutations
            
            unique_true = np.unique(y)
            unique_pred = np.unique(predictions)
            
            best_accuracy = 0
            for perm in permutations(unique_pred):
                if len(perm) != len(unique_true):
                    continue
                
                mapping = dict(zip(perm, unique_true))
                mapped_pred = np.array([mapping.get(label, -1) for label in predictions])
                accuracy = np.mean(mapped_pred == y)
                best_accuracy = max(best_accuracy, accuracy)
            
            return best_accuracy
        except:
            return None


def create_performance_report(benchmark_results: Dict[str, AlgorithmBenchmark], 
                            output_file: str = "performance_report.txt") -> None:
    """
    Create a comprehensive performance report from benchmark results.
    
    Args:
        benchmark_results: Dictionary mapping algorithm names to benchmark results
        output_file: Output file for the report
    """
    with open(output_file, 'w') as f:
        f.write("AI ALGORITHMS COLLECTION - PERFORMANCE REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("This report contains performance benchmarks for all algorithms\n")
        f.write("in the AI algorithms collection. Metrics include execution time,\n")
        f.write("memory usage, and accuracy across different dataset sizes.\n\n")
        
        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 30 + "\n")
        
        for alg_name, benchmark in benchmark_results.items():
            results = benchmark.results
            f.write(f"\n{alg_name}:\n")
            
            if results['execution_times']:
                avg_time = np.mean(results['execution_times'])
                f.write(f"  Average Execution Time: {avg_time:.4f} seconds\n")
            
            if results['memory_usage']:
                avg_memory = np.mean(results['memory_usage'])
                f.write(f"  Average Memory Usage: {avg_memory:.2f} MB\n")
            
            if results['accuracy_scores']:
                avg_accuracy = np.mean(results['accuracy_scores'])
                f.write(f"  Average Accuracy: {avg_accuracy:.4f}\n")
        
        # Detailed results
        f.write("\n\nDETAILED RESULTS\n")
        f.write("-" * 30 + "\n")
        
        for alg_name, benchmark in benchmark_results.items():
            results = benchmark.results
            f.write(f"\n{alg_name} - Detailed Performance:\n")
            f.write("Dataset Size | Exec Time (s) | Memory (MB) | Accuracy\n")
            f.write("-" * 50 + "\n")
            
            for i in range(len(results['dataset_sizes'])):
                size = results['dataset_sizes'][i]
                time_val = results['execution_times'][i] if i < len(results['execution_times']) else 'N/A'
                memory_val = results['memory_usage'][i] if i < len(results['memory_usage']) else 'N/A'
                accuracy_val = results['accuracy_scores'][i] if i < len(results['accuracy_scores']) else 'N/A'
                
                f.write(f"{size:11d} | {time_val:>11.4f} | {memory_val:>9.2f} | {accuracy_val:>8.4f}\n"
                       if isinstance(time_val, (int, float)) else
                       f"{size:11d} | {str(time_val):>11} | {str(memory_val):>9} | {str(accuracy_val):>8}\n")
        
        # Performance rankings
        f.write("\n\nPERFORMANCE RANKINGS\n")
        f.write("-" * 30 + "\n")
        
        # Rank by execution time
        time_rankings = []
        for alg_name, benchmark in benchmark_results.items():
            if benchmark.results['execution_times']:
                avg_time = np.mean(benchmark.results['execution_times'])
                time_rankings.append((alg_name, avg_time))
        
        time_rankings.sort(key=lambda x: x[1])
        
        f.write("\nFastest Algorithms (by average execution time):\n")
        for i, (alg_name, avg_time) in enumerate(time_rankings):
            f.write(f"{i+1}. {alg_name}: {avg_time:.4f} seconds\n")
        
        # Rank by accuracy
        accuracy_rankings = []
        for alg_name, benchmark in benchmark_results.items():
            if benchmark.results['accuracy_scores']:
                avg_accuracy = np.mean(benchmark.results['accuracy_scores'])
                accuracy_rankings.append((alg_name, avg_accuracy))
        
        accuracy_rankings.sort(key=lambda x: x[1], reverse=True)
        
        f.write("\nMost Accurate Algorithms:\n")
        for i, (alg_name, avg_accuracy) in enumerate(accuracy_rankings):
            f.write(f"{i+1}. {alg_name}: {avg_accuracy:.4f}\n")
        
        f.write(f"\n\nReport generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def run_comprehensive_benchmarks():
    """Run comprehensive benchmarks on all algorithm categories."""
    print("Running Comprehensive AI Algorithms Benchmarks")
    print("=" * 60)
    
    benchmark_results = {}
    
    # This is a placeholder - actual implementation would import and test real algorithms
    print("Note: This is a framework for benchmarking.")
    print("To run actual benchmarks, ensure all algorithm modules are properly imported.")
    print("The benchmarking framework is designed to work with:")
    print("- Machine Learning algorithms (LinearRegression, LogisticRegression, etc.)")
    print("- Neural Networks (MultiLayer Perceptron, RBF networks)")
    print("- Optimization algorithms (PSO, Simulated Annealing, etc.)")
    print("- Clustering algorithms (K-Means, DBSCAN, Hierarchical)")
    print("- Pathfinding algorithms (A*, variations)")
    print("- Game algorithms (Minimax with Alpha-Beta pruning)")
    
    # Example benchmark structure
    sample_benchmark = AlgorithmBenchmark("Sample Algorithm")
    sample_benchmark.results = {
        'execution_times': [0.001, 0.005, 0.012, 0.025],
        'memory_usage': [1.2, 2.8, 5.1, 10.3],
        'accuracy_scores': [0.95, 0.93, 0.94, 0.92],
        'dataset_sizes': [100, 500, 1000, 2000],
        'additional_metrics': {}
    }
    
    benchmark_results["Sample Algorithm"] = sample_benchmark
    
    # Create performance report
    create_performance_report(benchmark_results, "ai_algorithms_performance_report.txt")
    
    print("\nBenchmarking framework ready!")
    print("Performance report template created: ai_algorithms_performance_report.txt")
    
    return benchmark_results


if __name__ == "__main__":
    # Run benchmarks
    results = run_comprehensive_benchmarks()
    
    print("\nBenchmarking completed!")
    print("Check the generated performance report for detailed results.")
