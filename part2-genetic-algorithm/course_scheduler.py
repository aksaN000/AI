with open('input.txt') as f:
    N, T = map(int, f.readline().split())
    courses = [i.strip('\n') for i in f.readlines()]

def fitness(chromosome):
    d = {}
    slot = 1
    temp = 0
    li = ''
    total_overlapping, total_consistency = 0, 0
    course_count = [0] * N

    for i in range(len(chromosome) + 1):
        if temp < N:
            li += chromosome[i]
            temp += 1
        else:
            d[slot] = li
            slot += 1
            temp = 1
            if i != len(chromosome):
                li = chromosome[i]

    for slot_schedule in d.values():
        courses_scheduled = 0
        for idx, j in enumerate(slot_schedule):
            if j == '1':
                courses_scheduled += 1
                course_count[idx] += 1
        total_overlapping += abs(courses_scheduled - 1)

    for count in course_count:
        total_consistency += abs(count - 1)
    print(f'Chromosome splitted by slots: {d}')

    return ((total_overlapping + total_consistency)*-1)


import random
def generate_chromosome(N, T):
    chromosome = ''.join(random.choice('01') for _ in range(N * T))
    return chromosome
def generate_population(N, T):
    population = []
    for i in range(10):
        chromosome = generate_chromosome(N, T)
        population.append(chromosome)
    return population
def random_parents(population):
    return random.sample(population, 2)


def single_point_crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    offspring1 = parent1[:point] + parent2[point:]
    offspring2 = parent2[:point] + parent1[point:]
    return offspring1, offspring2

def mutation(chromosome, mutation_rate):
    mutated = ''
    for gene in chromosome:
        if random.random() < mutation_rate: #random.random generates any float from [0,1]
            if gene == '0':
                mutated += '1'
            else:
                mutated += '0'
        else:
            mutated += gene
    return mutated


def genetic_algorithm(N, T, population_size, mutation_rate, max_iterations):
    population = generate_population(N, T)

    for i in range(max_iterations):
        population = sorted(population, key=lambda chromo: fitness(chromo), reverse=True)
        best_chromosome = population[0]
        best_fitness = fitness(best_chromosome)
        print(f"Iteration {i + 1}: Best Fitness = {best_fitness}")

        if best_fitness == 0:
            break

        new_population = []
        while len(new_population) <= population_size:
            parent1, parent2 = random_parents(population)
            offspring1, offspring2 = single_point_crossover(parent1, parent2)
            offspring1 = mutation(offspring1, mutation_rate)
            offspring2 = mutation(offspring2, mutation_rate)
            new_population.extend([offspring1, offspring2])

        population = new_population[:population_size]

    else:
        print(f"Maximum number of iterations reached. So far best chromosome is {best_chromosome}")

    return best_chromosome


def two_point_crossover(parent1, parent2):
    length = len(parent1)
    point1 = random.randint(0, length - 2)
    point2 = random.randint(point1 + 1, length - 1)

    print(f"Parent 1: {parent1}")
    print(f"Parent 2: {parent2}")
    print(f"Chosen crossover points: {point1} and {point2} (Considering 1st index as 1)")

    offspring1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    offspring2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]

    return offspring1, offspring2


if __name__ == "__main__":
    population_size = 10
    mutation_rate = 0.1
    max_iterations = 100

    print("Running Course Scheduling Genetic Algorithm")
    best_chromosome = genetic_algorithm(N, T, population_size, mutation_rate, max_iterations)
    print(f"Best Chromosome Found: {best_chromosome}")

    print("\nDemonstrating Two-Point Crossover:")
    initial_population = generate_population(N, T)
    parent1, parent2 = random_parents(initial_population)

    offspring1, offspring2 = two_point_crossover(parent1, parent2)
    print(f"Offspring 1: {offspring1}")
    print(f"Offspring 2: {offspring2}")
