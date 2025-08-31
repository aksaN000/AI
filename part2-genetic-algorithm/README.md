# Genetic Algorithm for Course Scheduling

This implementation uses genetic algorithms to solve course scheduling optimization problems, finding optimal arrangements of courses across time slots while minimizing conflicts.

## Overview

The genetic algorithm evolves a population of candidate solutions (chromosomes) representing course schedules. Each chromosome encodes which courses are scheduled in which time slots using binary representation.

## Problem Definition

**Goal**: Schedule N courses across T time slots such that:
- Each course is scheduled exactly once
- Each time slot has exactly one course
- Minimize overlapping and consistency violations

## Algorithm Components

### Chromosome Representation
- Binary string of length N × T
- Each substring of length N represents one time slot
- '1' indicates a course is scheduled, '0' indicates it's not

Example for 3 courses, 3 slots:
```
"100010001" = Course1 in Slot1, Course2 in Slot2, Course3 in Slot3
```

### Fitness Function
Evaluates chromosome quality by penalizing:
- **Overlapping**: Multiple courses in same time slot
- **Consistency**: Courses scheduled multiple times or not at all

```python
fitness = -1 × (total_overlapping + total_consistency)
```
Higher fitness values indicate better solutions (fitness = 0 is optimal).

### Genetic Operators

#### Selection
- **Random Parents**: Randomly selects two parents from population
- Population sorted by fitness for better selection pressure

#### Crossover Operations

1. **Single-Point Crossover**
   - Choose random crossover point
   - Exchange genetic material after the point
   - Creates two offspring from two parents

2. **Two-Point Crossover**
   - Choose two random crossover points
   - Exchange middle segment between parents
   - Provides more diverse genetic combinations

#### Mutation
- **Bit Flip Mutation**: Randomly flip bits with given probability
- **Mutation Rate**: Configurable (default: 0.1)
- Maintains genetic diversity in population

## Parameters

```python
population_size = 10        # Number of chromosomes in population
mutation_rate = 0.1         # Probability of gene mutation
max_iterations = 100        # Maximum generations to evolve
```

## Usage

### Input Format (`input.txt`)
```
N T
Course1Name
Course2Name
...
CourseNName
```

Example:
```
3 3
CSE110
MAT110
PHY112
```

### Running the Algorithm
```python
python course_scheduler.py
```

### Output
The program displays:
- Iteration progress with fitness scores
- Best chromosome found
- Demonstration of crossover operations
- Final schedule arrangement

## Algorithm Flow

1. **Initialize**: Generate random population
2. **Evaluate**: Calculate fitness for each chromosome
3. **Select**: Choose parents for reproduction
4. **Reproduce**: Apply crossover and mutation
5. **Replace**: Form new generation
6. **Repeat**: Until optimal solution found or max iterations reached

## Convergence Criteria

The algorithm stops when:
- Perfect solution found (fitness = 0)
- Maximum iterations reached
- Population converges to local optimum

## Performance Analysis

### Time Complexity
- **Per Generation**: O(P × N × T) where P is population size
- **Total**: O(G × P × N × T) where G is number of generations

### Space Complexity
- **Memory Usage**: O(P × N × T) for storing population
- **Additional**: O(N × T) for temporary operations

## Optimization Strategies

1. **Elitism**: Preserve best solutions across generations
2. **Tournament Selection**: Better parent selection method
3. **Adaptive Mutation**: Adjust mutation rate based on diversity
4. **Constraint Handling**: Specialized operators for valid schedules

## Example Results

For 3 courses and 3 time slots:
```
Iteration 1: Best Fitness = -4
Iteration 15: Best Fitness = -2
Iteration 23: Best Fitness = 0
Best Chromosome Found: 100010001

Schedule:
Slot 1: CSE110
Slot 2: MAT110  
Slot 3: PHY112
```

## Applications

This algorithm can be extended for:
- University course timetabling
- Employee shift scheduling
- Resource allocation problems
- Task assignment optimization

## Limitations

- May converge to local optima
- Performance depends on problem size
- Requires tuning of genetic parameters
- No guarantee of global optimum

## Future Enhancements

- Multi-objective optimization
- Constraint satisfaction techniques
- Hybrid algorithms with local search
- Parallel genetic algorithm implementation
