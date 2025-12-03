import numpy as np
from base.TestFunctions import Sphere, Rastrigin, Rosenbrock, Griewank
from algorithms.EvolutionStrategy import ES 

sphere_func = Sphere()
rastrigin_func = Rastrigin()
rosenbrock_func = Rosenbrock()
griewank_func = Griewank()

es = ES(mu=10, lam=40, dim=2, max_iter=50, low=-5, high=5, func=griewank_func)
best = es.run()

print("\n--- TEST ZAKOŃCZONY ---")
print("Najlepszy Fitness:", best.fitness) 
print("Genom:", best.genom)