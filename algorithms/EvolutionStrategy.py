import numpy as np
from base.BaseAlgorithm import BaseAlgorithm, Individual

class ES(BaseAlgorithm):
    def __init__(self, mu, lam, dim, max_iter, low, high, func):
        super().__init__(func, dim, max_iter, low, high)
        self.mu = mu
        self.lam = lam
        self.tau = 1 / np.sqrt(2 * np.sqrt(dim))

    def initialize(self):
        for _ in range(self.mu):
            genom = np.random.uniform(self.low, self.high, self.dim)
            sigma = np.random.uniform(0.1, 1.0)
            self.population.append(Individual(genom, sigma))

    def recombine(self):
        parents = np.random.choice(self.population, 2, replace=False)

        child_genom = (parents[0].genom + parents[1].genom) / 2
        child_sigma = (parents[0].sigma + parents[1].sigma) / 2

        return Individual(child_genom.copy(), child_sigma)

    def mutate(self, ind):
        ind.sigma *= np.exp(self.tau * np.random.randn())

        ind.genom = ind.genom + ind.sigma * np.random.randn(self.dim)

    def generate_children(self):
        children = []

        for _ in range(self.lam):
            child = self.recombine()
            self.mutate(child)
            self.evaluate(child)
            children.append(child)

        return children

    def select(self, children):
        combined = self.population + children
        combined.sort(key=lambda x: x.fitness)
        self.population = combined[:self.mu]

    def run_with_progress(self, progress_bar=None): 
            self.initialize()
            self.population_history = []
            for ind in self.population:
                self.evaluate(ind)

            current_generation_data = []
            for ind in self.population:
                current_generation_data.append({
                    "genom": ind.genom.copy(),
                    "fitness": ind.fitness
                })
            self.population_history.append(current_generation_data)
                
            if progress_bar:
                progress_bar.progress(1, text="Inicjalizacja zakończona...")

            for gen in range(self.max_iter):
                children = self.generate_children()
                self.select(children)
                best = self.population[0]
                self.history.append(best.copy())

                current_generation_data = []
                for ind in self.population:
                    current_generation_data.append({
                        "genom": ind.genom.copy(),
                        "fitness": ind.fitness
                    })
                self.population_history.append(current_generation_data)

                if progress_bar:
                    percent_complete = (gen + 1) / self.max_iter
                    progress_bar.progress(percent_complete, text=f"Generacja {gen+1}/{self.max_iter} | Best: {best.fitness:.4e}")

            self.best_individual = self.population[0]
            
            # Ustawienie paska postępu na 100%
            if progress_bar:
                progress_bar.progress(100, text="Optymalizacja zakończona.")
                
            return self.best_individual

    # WAŻNE: Wymagana jest też oryginalna metoda run, jeśli jest abstrakcyjna
    def run(self):
        return self.run_with_progress(progress_bar=None)


