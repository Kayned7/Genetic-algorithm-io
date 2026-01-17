import numpy as np
from base.BaseAlgorithm import BaseAlgorithm, Individual


class GA(BaseAlgorithm):
    def __init__(self, func, dim, max_iter, low, high, pop_size=50, mutation_prob=0.1, crossover_prob=0.8,
                 tournament_size=3):
        super().__init__(func, dim, max_iter, low, high)

        self.pop_size = pop_size
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.tournament_size = tournament_size

    def initialize(self):
        self.population = []
        for _ in range(self.pop_size):
            genom = np.random.uniform(self.low, self.high, self.dim)
            sigma = 0.5
            self.population.append(Individual(genom, sigma))

    def tournament_selection(self):
        candidates = np.random.choice(self.population, self.tournament_size, replace=False)
        best = min(candidates, key=lambda x: x.fitness)
        return best

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_prob:
            alpha = np.random.rand()
            child_genom = alpha * parent1.genom + (1 - alpha) * parent2.genom

            child_sigma = (parent1.sigma + parent2.sigma) / 2
            return Individual(child_genom, child_sigma)
        else:
            return parent1.copy()

    def mutate(self, individual):
        if np.random.rand() < self.mutation_prob:
            noise = np.random.normal(0, 1.0, self.dim)
            scale = (self.high - self.low) * 0.05

            individual.genom += noise * scale

            individual.genom = np.clip(individual.genom, self.low, self.high)

    def run_with_progress(self, progress_bar=None):
        self.initialize()
        self.population_history = []
        for ind in self.population:
            self.evaluate(ind)

        self.population.sort(key=lambda ind: ind.fitness)
        self.best_individual = self.population[0]
        self.history.append(self.best_individual.copy())

        current_generation_data = []
        for ind in self.population:
            current_generation_data.append({
                "genom": ind.genom.copy(),
                "fitness": ind.fitness
            })
        self.population_history.append(current_generation_data)

        if progress_bar:
            progress_bar.progress(0, text="Inicjalizacja GA...")

        for gen in range(self.max_iter):
            new_population = []

            new_population.append(self.population[0].copy())

            while len(new_population) < self.pop_size:
                p1 = self.tournament_selection()
                p2 = self.tournament_selection()

                child = self.crossover(p1, p2)

                self.mutate(child)

                self.evaluate(child)

                new_population.append(child)

            self.population = new_population
            self.population.sort(key=lambda ind: ind.fitness)

            if self.population[0].fitness < self.best_individual.fitness:
                self.best_individual = self.population[0].copy()

            self.history.append(self.population[0].copy())

            current_generation_data = []
            for ind in self.population:
                current_generation_data.append({
                    "genom": ind.genom.copy(),
                    "fitness": ind.fitness
                })
            self.population_history.append(current_generation_data)

            if progress_bar:
                percent_complete = (gen + 1) / self.max_iter
                progress_bar.progress(percent_complete,
                                      text=f"GA Generacja {gen + 1}/{self.max_iter} | Best: {self.best_individual.fitness:.4e}")

        if progress_bar:
            progress_bar.progress(100, text="Optymalizacja GA zakończona.")

        return self.best_individual

    def run(self):
        """Wersja bez paska postępu (wymagana przez klasę abstrakcyjną)."""
        return self.run_with_progress(progress_bar=None)