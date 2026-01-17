import numpy as np
from base.BaseAlgorithm import BaseAlgorithm, Individual


class PSO(BaseAlgorithm):
    def __init__(self, func, dim, max_iter, low, high, pop_size=30, w=0.5, c1=1.5, c2=1.5):
        """
        w  - waga inercji (jak bardzo cząstka chce zachować swój pęd)
        c1 - współczynnik kognitywny (jak bardzo ufa swojej pamięci)
        c2 - współczynnik socjalny (jak bardzo ufa liderowi stada)
        """
        super().__init__(func, dim, max_iter, low, high)
        self.pop_size = pop_size
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def initialize(self):
        self.population = []
        for _ in range(self.pop_size):
            # 1. Inicjalizacja pozycji (genomu)
            genom = np.random.uniform(self.low, self.high, self.dim)

            # Parametr sigma w PSO nie jest używany, wpisujemy 0 lub None,
            # ale tworzymy obiekt Individual dla zgodności z BaseAlgorithm
            ind = Individual(genom, sigma=0.0)

            # 2. Inicjalizacja atrybutów specyficznych dla PSO (doklejamy je dynamicznie)
            # Prędkość początkowa (zazwyczaj losowa w niewielkim zakresie)
            ind.velocity = np.random.uniform(-1, 1, self.dim)

            # Pamięć najlepszej pozycji (p_best) - na początku to pozycja startowa
            ind.local_best_genom = ind.genom.copy()
            ind.local_best_fitness = float('inf')  # Zakładamy minimalizację

            self.population.append(ind)

    def update_particle(self, ind):
        """Aktualizuje prędkość i pozycję pojedynczej cząstki."""
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)

        # Wzór na prędkość: v(t+1) = w*v(t) + c1*r1*(p_best - x) + c2*r2*(g_best - x)
        cognitive_velocity = self.c1 * r1 * (ind.local_best_genom - ind.genom)
        social_velocity = self.c2 * r2 * (self.best_individual.genom - ind.genom)

        ind.velocity = self.w * ind.velocity + cognitive_velocity + social_velocity

        # Aktualizacja pozycji: x(t+1) = x(t) + v(t+1)
        ind.genom = ind.genom + ind.velocity

        # Ograniczenie do granic (ściany pudełka)
        # Jeśli cząstka wyleci, przycinamy ją do krawędzi (można też odbijać wektor prędkości)
        ind.genom = np.clip(ind.genom, self.low, self.high)

    def run_with_progress(self, progress_bar=None):
        self.initialize()
        self.population_history = []
        # Pierwsza ocena populacji, aby ustalić punkty startowe
        for ind in self.population:
            self.evaluate(ind)
            # Na starcie obecna pozycja jest najlepszą znaną dla cząstki
            ind.local_best_fitness = ind.fitness
            ind.local_best_genom = ind.genom.copy()

        # Znalezienie globalnego lidera (g_best)
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

        # Główna pętla ewolucji/ruchu
        for gen in range(self.max_iter):
            for ind in self.population:
                # 1. Ruch cząstki
                self.update_particle(ind)

                # 2. Ocena nowej pozycji
                self.evaluate(ind)

                # 3. Aktualizacja pamięci osobistej (p_best)
                if ind.fitness < ind.local_best_fitness:
                    ind.local_best_fitness = ind.fitness
                    ind.local_best_genom = ind.genom.copy()

                # 4. Aktualizacja globalnego lidera (g_best)
                # Sprawdzamy na bieżąco (asynchronicznie) lub po całej epoce.
                # Tutaj sprawdzamy od razu dla szybszej zbieżności.
                if ind.fitness < self.best_individual.fitness:
                    self.best_individual = ind.copy()  # Ważne: kopia, bo ind się zaraz zmieni

            # Zapis historii
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
        """Wersja wymagana przez klasę abstrakcyjną (bez GUI)"""
        return self.run_with_progress(progress_bar=None)