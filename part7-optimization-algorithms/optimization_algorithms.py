import numpy as np
import random
import math
from typing import List, Tuple, Callable, Optional, Dict, Any
from abc import ABC, abstractmethod


class OptimizationAlgorithm(ABC):
    """Abstract base class for optimization algorithms."""
    
    @abstractmethod
    def optimize(self, objective_function: Callable, bounds: List[Tuple[float, float]], 
                max_iterations: int = 1000) -> Tuple[np.ndarray, float]:
        """
        Optimize the given objective function.
        
        Args:
            objective_function: Function to minimize
            bounds: List of (min, max) bounds for each dimension
            max_iterations: Maximum number of iterations
            
        Returns:
            best_solution: Best solution found
            best_fitness: Best fitness value
        """
        pass


class ParticleSwarmOptimization(OptimizationAlgorithm):
    """
    Particle Swarm Optimization (PSO) algorithm.
    Bio-inspired optimization based on social behavior of bird flocking.
    """
    
    def __init__(self, n_particles: int = 30, w: float = 0.729, c1: float = 1.494, c2: float = 1.494):
        """
        Initialize PSO parameters.
        
        Args:
            n_particles: Number of particles in the swarm
            w: Inertia weight
            c1: Cognitive acceleration coefficient
            c2: Social acceleration coefficient
        """
        self.n_particles = n_particles
        self.w = w  # Inertia weight
        self.c1 = c1  # Personal best weight
        self.c2 = c2  # Global best weight
        
    def optimize(self, objective_function: Callable, bounds: List[Tuple[float, float]], 
                max_iterations: int = 1000) -> Tuple[np.ndarray, float]:
        """
        Optimize using Particle Swarm Optimization.
        
        Args:
            objective_function: Function to minimize
            bounds: List of (min, max) bounds for each dimension
            max_iterations: Maximum number of iterations
            
        Returns:
            best_solution: Best solution found
            best_fitness: Best fitness value
        """
        n_dimensions = len(bounds)
        
        # Initialize particles
        particles = []
        velocities = []
        personal_best_positions = []
        personal_best_scores = []
        
        for _ in range(self.n_particles):
            # Random initial position within bounds
            position = np.array([random.uniform(bound[0], bound[1]) for bound in bounds])
            velocity = np.array([random.uniform(-1, 1) for _ in range(n_dimensions)])
            
            particles.append(position)
            velocities.append(velocity)
            personal_best_positions.append(position.copy())
            personal_best_scores.append(objective_function(position))
        
        # Find initial global best
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)].copy()
        global_best_score = min(personal_best_scores)
        
        # Optimization loop
        for iteration in range(max_iterations):
            for i in range(self.n_particles):
                # Update velocity
                r1, r2 = random.random(), random.random()
                
                cognitive_velocity = self.c1 * r1 * (personal_best_positions[i] - particles[i])
                social_velocity = self.c2 * r2 * (global_best_position - particles[i])
                
                velocities[i] = (self.w * velocities[i] + 
                               cognitive_velocity + social_velocity)
                
                # Update position
                particles[i] += velocities[i]
                
                # Apply bounds
                for j in range(n_dimensions):
                    particles[i][j] = np.clip(particles[i][j], bounds[j][0], bounds[j][1])
                
                # Evaluate fitness
                fitness = objective_function(particles[i])
                
                # Update personal best
                if fitness < personal_best_scores[i]:
                    personal_best_scores[i] = fitness
                    personal_best_positions[i] = particles[i].copy()
                    
                    # Update global best
                    if fitness < global_best_score:
                        global_best_score = fitness
                        global_best_position = particles[i].copy()
            
            # Optional: Print progress
            if iteration % 100 == 0:
                print(f"PSO Iteration {iteration}: Best fitness = {global_best_score:.6f}")
        
        return global_best_position, global_best_score


class SimulatedAnnealing(OptimizationAlgorithm):
    """
    Simulated Annealing optimization algorithm.
    Probabilistic technique that mimics the annealing process in metallurgy.
    """
    
    def __init__(self, initial_temperature: float = 100.0, cooling_rate: float = 0.95, 
                 min_temperature: float = 1e-8):
        """
        Initialize Simulated Annealing parameters.
        
        Args:
            initial_temperature: Starting temperature
            cooling_rate: Rate at which temperature decreases
            min_temperature: Minimum temperature threshold
        """
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        
    def optimize(self, objective_function: Callable, bounds: List[Tuple[float, float]], 
                max_iterations: int = 1000) -> Tuple[np.ndarray, float]:
        """
        Optimize using Simulated Annealing.
        
        Args:
            objective_function: Function to minimize
            bounds: List of (min, max) bounds for each dimension
            max_iterations: Maximum number of iterations
            
        Returns:
            best_solution: Best solution found
            best_fitness: Best fitness value
        """
        n_dimensions = len(bounds)
        
        # Initialize random solution
        current_solution = np.array([random.uniform(bound[0], bound[1]) for bound in bounds])
        current_fitness = objective_function(current_solution)
        
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        
        temperature = self.initial_temperature
        
        for iteration in range(max_iterations):
            # Generate neighbor solution
            step_size = temperature / self.initial_temperature  # Adaptive step size
            neighbor_solution = current_solution + np.random.normal(0, step_size, n_dimensions)
            
            # Apply bounds
            for i in range(n_dimensions):
                neighbor_solution[i] = np.clip(neighbor_solution[i], bounds[i][0], bounds[i][1])
            
            neighbor_fitness = objective_function(neighbor_solution)
            
            # Accept or reject the neighbor
            if neighbor_fitness < current_fitness:
                # Accept better solution
                current_solution = neighbor_solution
                current_fitness = neighbor_fitness
                
                # Update best solution
                if neighbor_fitness < best_fitness:
                    best_solution = neighbor_solution.copy()
                    best_fitness = neighbor_fitness
            else:
                # Accept worse solution with probability
                delta = neighbor_fitness - current_fitness
                probability = math.exp(-delta / temperature)
                
                if random.random() < probability:
                    current_solution = neighbor_solution
                    current_fitness = neighbor_fitness
            
            # Cool down
            temperature *= self.cooling_rate
            
            # Check stopping criterion
            if temperature < self.min_temperature:
                break
            
            # Optional: Print progress
            if iteration % 100 == 0:
                print(f"SA Iteration {iteration}: Best fitness = {best_fitness:.6f}, Temperature = {temperature:.6f}")
        
        return best_solution, best_fitness


class DifferentialEvolution(OptimizationAlgorithm):
    """
    Differential Evolution (DE) optimization algorithm.
    Evolutionary algorithm that uses vector differences for mutation.
    """
    
    def __init__(self, population_size: int = 50, F: float = 0.8, CR: float = 0.9):
        """
        Initialize Differential Evolution parameters.
        
        Args:
            population_size: Number of individuals in population
            F: Mutation factor (0 < F <= 2)
            CR: Crossover probability (0 <= CR <= 1)
        """
        self.population_size = population_size
        self.F = F  # Mutation factor
        self.CR = CR  # Crossover probability
        
    def optimize(self, objective_function: Callable, bounds: List[Tuple[float, float]], 
                max_iterations: int = 1000) -> Tuple[np.ndarray, float]:
        """
        Optimize using Differential Evolution.
        
        Args:
            objective_function: Function to minimize
            bounds: List of (min, max) bounds for each dimension
            max_iterations: Maximum number of iterations
            
        Returns:
            best_solution: Best solution found
            best_fitness: Best fitness value
        """
        n_dimensions = len(bounds)
        
        # Initialize population
        population = []
        fitness_values = []
        
        for _ in range(self.population_size):
            individual = np.array([random.uniform(bound[0], bound[1]) for bound in bounds])
            population.append(individual)
            fitness_values.append(objective_function(individual))
        
        # Track best solution
        best_index = np.argmin(fitness_values)
        best_solution = population[best_index].copy()
        best_fitness = fitness_values[best_index]
        
        # Evolution loop
        for iteration in range(max_iterations):
            new_population = []
            new_fitness_values = []
            
            for i in range(self.population_size):
                # Mutation: select three random individuals (different from current)
                candidates = [j for j in range(self.population_size) if j != i]
                a, b, c = random.sample(candidates, 3)
                
                # Mutant vector
                mutant = population[a] + self.F * (population[b] - population[c])
                
                # Apply bounds to mutant
                for j in range(n_dimensions):
                    mutant[j] = np.clip(mutant[j], bounds[j][0], bounds[j][1])
                
                # Crossover
                trial = population[i].copy()
                j_rand = random.randint(0, n_dimensions - 1)  # Ensure at least one parameter is taken from mutant
                
                for j in range(n_dimensions):
                    if random.random() < self.CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Selection
                trial_fitness = objective_function(trial)
                
                if trial_fitness <= fitness_values[i]:
                    new_population.append(trial)
                    new_fitness_values.append(trial_fitness)
                    
                    # Update best solution
                    if trial_fitness < best_fitness:
                        best_solution = trial.copy()
                        best_fitness = trial_fitness
                else:
                    new_population.append(population[i])
                    new_fitness_values.append(fitness_values[i])
            
            population = new_population
            fitness_values = new_fitness_values
            
            # Optional: Print progress
            if iteration % 100 == 0:
                print(f"DE Iteration {iteration}: Best fitness = {best_fitness:.6f}")
        
        return best_solution, best_fitness


class AntColonyOptimization:
    """
    Ant Colony Optimization (ACO) for continuous optimization problems.
    Adapted version of ACO for real-valued optimization.
    """
    
    def __init__(self, n_ants: int = 50, n_archive: int = 50, q: float = 0.01, 
                 xi: float = 0.85, sigma_init: float = 0.1):
        """
        Initialize ACO parameters.
        
        Args:
            n_ants: Number of ants
            n_archive: Size of solution archive
            q: Locality parameter
            xi: Speed of convergence
            sigma_init: Initial standard deviation
        """
        self.n_ants = n_ants
        self.n_archive = n_archive
        self.q = q
        self.xi = xi
        self.sigma_init = sigma_init
        
    def optimize(self, objective_function: Callable, bounds: List[Tuple[float, float]], 
                max_iterations: int = 1000) -> Tuple[np.ndarray, float]:
        """
        Optimize using Ant Colony Optimization.
        
        Args:
            objective_function: Function to minimize
            bounds: List of (min, max) bounds for each dimension
            max_iterations: Maximum number of iterations
            
        Returns:
            best_solution: Best solution found
            best_fitness: Best fitness value
        """
        n_dimensions = len(bounds)
        
        # Initialize solution archive
        archive = []
        archive_fitness = []
        
        # Generate initial solutions
        for _ in range(self.n_archive):
            solution = np.array([random.uniform(bound[0], bound[1]) for bound in bounds])
            fitness = objective_function(solution)
            archive.append(solution)
            archive_fitness.append(fitness)
        
        # Sort archive by fitness
        sorted_indices = np.argsort(archive_fitness)
        archive = [archive[i] for i in sorted_indices]
        archive_fitness = [archive_fitness[i] for i in sorted_indices]
        
        best_solution = archive[0].copy()
        best_fitness = archive_fitness[0]
        
        # Main optimization loop
        for iteration in range(max_iterations):
            new_solutions = []
            new_fitness_values = []
            
            # Generate new solutions using ants
            for _ in range(self.n_ants):
                # Select solution from archive based on rank
                probabilities = [(self.q / (math.sqrt(2 * math.pi) * self.sigma_init)) * 
                               math.exp(-0.5 * ((i) / self.sigma_init) ** 2) 
                               for i in range(len(archive))]
                probabilities = np.array(probabilities)
                probabilities /= np.sum(probabilities)
                
                selected_index = np.random.choice(len(archive), p=probabilities)
                selected_solution = archive[selected_index]
                
                # Generate new solution around selected one
                sigma = self.xi ** iteration * self.sigma_init
                new_solution = selected_solution + np.random.normal(0, sigma, n_dimensions)
                
                # Apply bounds
                for j in range(n_dimensions):
                    new_solution[j] = np.clip(new_solution[j], bounds[j][0], bounds[j][1])
                
                fitness = objective_function(new_solution)
                new_solutions.append(new_solution)
                new_fitness_values.append(fitness)
            
            # Update archive
            all_solutions = archive + new_solutions
            all_fitness = archive_fitness + new_fitness_values
            
            # Sort and keep best solutions
            sorted_indices = np.argsort(all_fitness)
            archive = [all_solutions[i] for i in sorted_indices[:self.n_archive]]
            archive_fitness = [all_fitness[i] for i in sorted_indices[:self.n_archive]]
            
            # Update best solution
            if archive_fitness[0] < best_fitness:
                best_solution = archive[0].copy()
                best_fitness = archive_fitness[0]
            
            # Optional: Print progress
            if iteration % 100 == 0:
                print(f"ACO Iteration {iteration}: Best fitness = {best_fitness:.6f}")
        
        return best_solution, best_fitness


# Test functions for optimization algorithms
class TestFunctions:
    """Collection of standard test functions for optimization algorithms."""
    
    @staticmethod
    def sphere(x: np.ndarray) -> float:
        """Sphere function: f(x) = sum(x_i^2)"""
        return np.sum(x ** 2)
    
    @staticmethod
    def rosenbrock(x: np.ndarray) -> float:
        """Rosenbrock function: challenging optimization problem"""
        return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
    
    @staticmethod
    def rastrigin(x: np.ndarray) -> float:
        """Rastrigin function: highly multimodal function"""
        A = 10
        n = len(x)
        return A * n + np.sum(x ** 2 - A * np.cos(2 * math.pi * x))
    
    @staticmethod
    def ackley(x: np.ndarray) -> float:
        """Ackley function: multimodal with global minimum at origin"""
        a, b, c = 20, 0.2, 2 * math.pi
        n = len(x)
        return (-a * math.exp(-b * math.sqrt(np.sum(x ** 2) / n)) - 
                math.exp(np.sum(np.cos(c * x)) / n) + a + math.e)
    
    @staticmethod
    def griewank(x: np.ndarray) -> float:
        """Griewank function: multimodal with many local minima"""
        return (np.sum(x ** 2) / 4000 - 
                np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))) + 1)


def compare_optimizers(objective_function: Callable, bounds: List[Tuple[float, float]], 
                      function_name: str = "Unknown", max_iterations: int = 500) -> Dict[str, Any]:
    """
    Compare different optimization algorithms on a given function.
    
    Args:
        objective_function: Function to optimize
        bounds: Optimization bounds
        function_name: Name of the function for display
        max_iterations: Maximum iterations for each algorithm
        
    Returns:
        Dictionary with results for each algorithm
    """
    print(f"\nOptimizing {function_name} function")
    print("=" * 50)
    
    optimizers = {
        'PSO': ParticleSwarmOptimization(n_particles=30),
        'SA': SimulatedAnnealing(initial_temperature=100.0),
        'DE': DifferentialEvolution(population_size=30),
        'ACO': AntColonyOptimization(n_ants=30)
    }
    
    results = {}
    
    for name, optimizer in optimizers.items():
        print(f"\nRunning {name}...")
        best_solution, best_fitness = optimizer.optimize(
            objective_function, bounds, max_iterations
        )
        
        results[name] = {
            'solution': best_solution,
            'fitness': best_fitness
        }
        
        print(f"{name} Result: fitness = {best_fitness:.8f}")
        print(f"{name} Solution: {best_solution}")
    
    return results


if __name__ == "__main__":
    # Example usage and comparison of optimization algorithms
    print("Advanced Optimization Algorithms Demo")
    print("=" * 40)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Test on different benchmark functions
    test_functions = {
        'Sphere': (TestFunctions.sphere, [(-5.12, 5.12)] * 2),
        'Rosenbrock': (TestFunctions.rosenbrock, [(-5, 10)] * 2),
        'Rastrigin': (TestFunctions.rastrigin, [(-5.12, 5.12)] * 2),
        'Ackley': (TestFunctions.ackley, [(-32.768, 32.768)] * 2)
    }
    
    all_results = {}
    
    for func_name, (func, bounds) in test_functions.items():
        results = compare_optimizers(func, bounds, func_name, max_iterations=300)
        all_results[func_name] = results
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    
    for func_name in test_functions.keys():
        print(f"\n{func_name} Function:")
        print("-" * 30)
        
        results = all_results[func_name]
        sorted_results = sorted(results.items(), key=lambda x: x[1]['fitness'])
        
        for i, (alg_name, result) in enumerate(sorted_results):
            print(f"{i+1}. {alg_name}: {result['fitness']:.8f}")
    
    print("\nOptimization algorithms comparison completed!")
    print("Note: Lower fitness values indicate better performance.")
