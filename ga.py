import numpy as np
from trader import Trader
import os
import pandas as pd
from tqdm import tqdm
class GeneticAlgorythm:
    def __init__(self, generations, mutation_rate, initial_mutation_strength=0.1, decay_rate=0.9,population_size=100, feature_size=7, output_size=3):
        # Hyperparameters
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.initial_mutation_strength = initial_mutation_strength
        self.decay_rate = decay_rate
        self.population_size = population_size
        self.feature_size = feature_size
        self.output_size = output_size
        self.current_generation = 0
        
        self.feature_size = feature_size
        self.output_size = output_size
        
        self.traders = []

        self.initialize_population()

    def initialize_population(self):
        # Initialize the population
        for _ in range(self.population_size):
            # window_size = np.random.randint(10, 60)
            # hidden_sizes = np.random.randint(1, 100, (np.random.randint(1, 5)))
            # self.traders.append(Trader(window_size, self.feature_size, hidden_sizes, self.output_size))
            window_size = 30
            hidden_sizes = [50, 40]
            self.traders.append(Trader(window_size, self.feature_size, hidden_sizes, self.output_size))

    def progress_day(self, dataset):
        while True:
            passed_minute = []
            for trader in self.traders:
                passed_minute.append(trader.pass_minute(dataset))
            if not any(passed_minute):
                break
    
    def mutate(self):
        mutation_strength = self.initial_mutation_strength * (self.decay_rate ** self.current_generation)
        for trader in self.traders:
            trader.mutate(mutation_rate=self.mutation_rate, mutation_strength=mutation_strength)
        self.current_generation += 1

    def genetic_algorithm(self, datasets):
        # Initialize the population
        population = self.traders
        
        for generation in tqdm(range(self.generations)):
            self.progress_day(datasets[generation % len(datasets)])
            fitness_scores = [trader.portfolio.profit() for trader in population]
            sorted_traders = sorted(zip(fitness_scores, population), key=lambda x: x[0], reverse=True)

            sorted_fitness_scores, sorted_traders = zip(*sorted_traders)
            new_population = []

            for i in range(int(self.population_size/10)):
                for _ in range(10):
                    new_population.append(sorted_traders[i].clown())
                
            self.traders = new_population
            self.mutate()

            print(f"Generation {generation}, Best Fitness {max(fitness_scores)}")

        self.progress_day(datasets[self.generations % len(datasets)])
        fitness_scores = [trader.portfolio.profit() for trader in population]
        
        return self.traders[np.argmax(fitness_scores)]

datasets = []
for filename in os.listdir('datasets'):
    if filename.endswith('.csv'):
        dataset = pd.read_csv(os.path.join('datasets', filename))
        dataset = dataset.iloc[:, 0:7]
        datasets.append(dataset)
ga = GeneticAlgorythm(generations=100, mutation_rate=0.8, initial_mutation_strength=0.2, decay_rate=0.9, population_size=100, feature_size=7, output_size=3)
best_network = ga.genetic_algorithm(datasets)
