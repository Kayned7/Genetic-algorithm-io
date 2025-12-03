import numpy as np

class Sphere:
    def __call__(self, x):
        return np.sum(x**2)
    
class Rastrigin:
    def __call__(self, x):
        return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    
class Rosenbrock:
    def __call__(self, x):
        result = 0.0
        for i in range(len(x) - 1):
            result += 100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        return result
    
class Griewank:
    def __call__(self, x):
        id = np.arange(1, len(x) + 1)
        sum_part = np.sum(x**2)
        cos_arg = x/np.sqrt(id)
        prod_part = np.prod(np.cos(cos_arg))
        return 1 + (1/4000) * sum_part - prod_part

class Beale:
    def __call__(self, x):
        term1 = (1.5 - x[0] + x[0]*x[1])**2
        term2 = (2.25 - x[0] + x[0]*(x[1])**2)**2
        term3 = (2.625 - x[0] + x[0]*(x[1])**3)**2
        return term1 + term2 + term3

class BukinN6:
    def __call__(self, x):
        term1 = 100 * np.sqrt(np.abs(x[1] - 0.01*(x[0])**2))
        term2 = 0.01 * np.abs(x[0] + 10)
        return term1 + term2