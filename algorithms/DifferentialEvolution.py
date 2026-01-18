import numpy as np
from base.BaseAlgorithm import BaseAlgorithm, Individual


class DifferentialEvolution(BaseAlgorithm):
    def __init__(self, func, dim, max_iter, low, high, pop_size=50, F=0.5, CR=0.7):
        super().__init__(func, dim, max_iter, low, high)
        self.pop_size = pop_size
        self.F = F
        self.CR = CR

    def initialize(self):
        self.population = []
        for _ in range(self.pop_size):
            genom = np.random.uniform(self.low, self.high, self.dim)
            sigma = 0.5
            self.population.append(Individual(genom, sigma))

    def run_with_progress(self, progress_bar=None):
        self.initialize()
        self.population_history = []

        for ind in self.population:
            self.evaluate(ind)


        self.population.sort(key=lambda ind: ind.fitness)
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
            progress_bar.progress(0, text="Inicjalizacja DE...")

        for gen in range(self.max_iter):
            new_population = []

            for i in range(self.pop_size):

                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a_idx, b_idx, c_idx = np.random.choice(idxs, 3, replace=False)

                a = self.population[a_idx].genom
                b = self.population[b_idx].genom
                c = self.population[c_idx].genom

                mutant_vector = a + self.F * (b - c)
                mutant_vector = np.clip(mutant_vector, self.low, self.high)

                target_vector = self.population[i].genom
                trial_vector = np.zeros(self.dim)

                j_rand = np.random.randint(self.dim)

                rand_vals = np.random.rand(self.dim)
                mask = (rand_vals < self.CR) | (np.arange(self.dim) == j_rand)

                trial_vector = np.where(mask, mutant_vector, target_vector)


                child = Individual(trial_vector, self.population[i].sigma)

                self.evaluate(child)

                if child.fitness <= self.population[i].fitness:
                    new_population.append(child)
                else:
                    new_population.append(self.population[i])

            self.population = new_population


            self.population.sort(key=lambda ind: ind.fitness)
            current_best = self.population[0]

            if current_best.fitness < self.best_individual.fitness:
                self.best_individual = current_best.copy()

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
                                      text=f"DE Generacja {gen + 1}/{self.max_iter} | Best: {self.best_individual.fitness:.4e}")

        if progress_bar:
            progress_bar.progress(100, text="Optymalizacja DE zakończona.")

        return self.best_individual

    def run(self):
        return self.run_with_progress(progress_bar=None)