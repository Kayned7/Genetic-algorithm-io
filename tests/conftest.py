# tests/conftest.py
import pytest
import numpy as np
from base.TestFunctions import Sphere, Beale
from algorithms.GeneticAlgorithm import GA
from algorithms.EvolutionStrategy import ES
from algorithms.PSO import PSO
from algorithms.DifferentialEvolution import DifferentialEvolution

@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(42)

@pytest.fixture
def sphere_function():
    return Sphere()

@pytest.fixture
def beale_function():
    return Beale()

@pytest.fixture
def default_params(sphere_function):
    return {
        "func": sphere_function,
        "dim": 2,
        "max_iter": 10,
        "low": -5.0,
        "high": 5.0
    }