from abc import ABC, abstractmethod

class Individual:
    def __init__(self, genom, sigma):
        self.genom = genom
        self.sigma = sigma
        self.fitness = None

    def copy(self):
        new_individual = Individual(self.genom.copy(), self.sigma)
        new_individual.fitness = self.fitness
        return new_individual

class BaseAlgorithm(ABC):
    def __init__(self, func, dim, max_iter, low, high):
        self.func = func
        self.dim = dim
        self.max_iter = max_iter
        self.low = low
        self.high = high
        self.best_individual = None
        self.population = []
        self.history = []

    @abstractmethod
    def initialize(self):
        pass
    
    @abstractmethod
    def run(self):
        pass

    def evaluate(self, individual):
        individual.fitness = self.func(individual.genom)

    def get_best(self):
        return self.best_individual.genom, self.best_individual.fitness