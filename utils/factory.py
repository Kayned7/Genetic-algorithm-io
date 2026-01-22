from algorithms.EvolutionStrategy import ES
from algorithms.GeneticAlgorithm import GA
from algorithms.PSO import PSO
from algorithms.DifferentialEvolution import DifferentialEvolution

def get_runner(alg_type, func, dim, low, high, max_iter, params):
    if low > high:
        low, high = high, low
    if alg_type == "Evolution Strategy (ES)":
        return ES(func=func, dim=dim, max_iter=max_iter, low=low, high=high, mu=params['mu'], lam=params['lam'])
    elif alg_type == "Genetic Algorithm (GA)":
        return GA(func=func, dim=dim, max_iter=max_iter, low=low, high=high, **params)
    elif alg_type == "Particle Swarm Optimization (PSO)":
        return PSO(func=func, dim=dim, max_iter=max_iter, low=low, high=high, **params)
    elif alg_type == "Differential Evolution (DE)":
        return DifferentialEvolution(func=func, dim=dim, max_iter=max_iter, low=low, high=high, **params)
    return None