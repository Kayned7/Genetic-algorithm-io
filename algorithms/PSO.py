import numpy as np
from base.BaseAlgorithm import BaseAlgorithm, Individual


class PSO(BaseAlgorithm):
    def __init__(self, func, dim, max_iter, low, high, pop_size=30, w=0.5, c1=1.5, c2=1.5):
        super().__init__(func, dim, max_iter, low, high)
        self.pop_size = pop_size
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def initialize(self):
        self.population = []
        for _ in range(self.pop_size):
            genom = np.random.uniform(self.low, self.high, self.dim)

            ind = Individual(genom, sigma=0.0)

            ind.velocity = np.random.uniform(-1, 1, self.dim)

            ind.local_best_genom = ind.genom.copy()
            ind.local_best_fitness = float('inf')

            self.population.append(ind)

    def update_particle(self, ind):
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)

        cognitive_velocity = self.c1 * r1 * (ind.local_best_genom - ind.genom)
        social_velocity = self.c2 * r2 * (self.best_individual.genom - ind.genom)

        ind.velocity = self.w * ind.velocity + cognitive_velocity + social_velocity

        ind.genom = ind.genom + ind.velocity
        ind.genom = np.clip(ind.genom, self.low, self.high)

    def run_with_progress(self, progress_bar=None):
        self.initialize()
        self.population_history = []
        for ind in self.population:
            self.evaluate(ind)
            ind.local_best_fitness = ind.fitness
            ind.local_best_genom = ind.genom.copy()

        self.population.sort(key=lambda x: x.fitness)
        self.best_individual = self.population[0].copy()
        self.history.append(self.best_individual.copy())

        current_generation_data = []
        for ind in self.population:
            current_generation_data.append({
                "genom": ind.genom.copy(),
                "fitness": ind.fitness
            })
        self.population_history.append(current_generation_data)

        if progress_bar:
            progress_bar.progress(0, text="Inicjalizacja roju PSO...")

        for gen in range(self.max_iter):
            for ind in self.population:

                self.update_particle(ind)

                self.evaluate(ind)

                if ind.fitness < ind.local_best_fitness:
                    ind.local_best_fitness = ind.fitness
                    ind.local_best_genom = ind.genom.copy()

                if ind.fitness < self.best_individual.fitness:
                    self.best_individual = ind.copy()

            self.history.append(self.best_individual.copy())

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
                                      text=f"PSO Iteracja {gen + 1}/{self.max_iter} | Best: {self.best_individual.fitness:.4e}")

        if progress_bar:
            progress_bar.progress(100, text="Optymalizacja rojem zakończona.")

        return self.best_individual

    def run(self):
        return self.run_with_progress(progress_bar=None)