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
        