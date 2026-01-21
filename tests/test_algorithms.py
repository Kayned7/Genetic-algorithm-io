import pytest
import numpy as np
from algorithms.GeneticAlgorithm import GA
from algorithms.EvolutionStrategy import ES
from algorithms.PSO import PSO
from algorithms.DifferentialEvolution import DifferentialEvolution
from base.BaseAlgorithm import Individual
from base.TestFunctions import Sphere


# fixtures

@pytest.fixture
def sphere_function():
    return Sphere()


@pytest.fixture
def common_params(sphere_function):
    return {
        "func": sphere_function,
        "dim": 2,
        "low": -5.0,
        "high": 5.0,
        "max_iter": 50
    }


# unity tests

def test_individual_creation():
    genom = np.array([1.0, 2.0])
    sigma = 0.5
    ind = Individual(genom, sigma)

    assert np.array_equal(ind.genom, genom)
    assert ind.sigma == sigma
    assert ind.fitness is None


def test_individual_copy():
    genom = np.array([1.0, 2.0])
    ind1 = Individual(genom, 0.5)
    ind2 = ind1.copy()

    ind2.genom[0] = 999.0

    assert ind1.genom[0] == 1.0, "Błąd: copy() nie stworzyło niezależnej kopii tablicy numpy!"
    assert ind2.genom[0] == 999.0


# integrity functions

class TestGA:
    def test_initialization(self, common_params):
        pop_size = 20
        ga = GA(pop_size=pop_size, **common_params)
        ga.initialize()

        assert len(ga.population) == pop_size
        assert len(ga.population[0].genom) == common_params['dim']
        # Sprawdzenie granic
        genom = ga.population[0].genom
        assert np.all(genom >= common_params['low'])
        assert np.all(genom <= common_params['high'])

    def test_convergence(self, common_params):
        params = common_params.copy()
        params['max_iter'] = 100

        ga = GA(pop_size=30, mutation_prob=0.1, crossover_prob=0.8, **params)
        best = ga.run()

        assert best.fitness < 1.0, f"GA nie zbiegł. Fitness: {best.fitness}"


class TestES:
    def test_initialization(self, common_params):
        mu, lam = 5, 20
        es = ES(mu=mu, lam=lam, dim=common_params['dim'], max_iter=common_params['max_iter'],
                low=common_params['low'], high=common_params['high'], func=common_params['func'])
        es.initialize()

        assert len(es.population) == mu
        assert hasattr(es.population[0], 'sigma')
        assert es.population[0].sigma > 0

    def test_convergence(self, common_params):
        es = ES(mu=10, lam=40, dim=common_params['dim'], max_iter=100,
                low=common_params['low'], high=common_params['high'], func=common_params['func'])
        best = es.run()

        assert best.fitness < 1.0, f"ES nie zbiegł. Fitness: {best.fitness}"


class TestPSO:
    def test_initialization(self, common_params):
        pop_size = 15
        pso = PSO(pop_size=pop_size, **common_params)
        pso.initialize()

        ind = pso.population[0]
        assert len(pso.population) == pop_size
        assert hasattr(ind, 'velocity')
        assert hasattr(ind, 'local_best_genom')
        assert ind.local_best_fitness == float('inf')

    @pytest.mark.parametrize("dim", [2, 5])
    def test_convergence_various_dims(self, sphere_function, dim):
        pso = PSO(func=sphere_function, dim=dim, max_iter=100, low=-5, high=5, pop_size=30)
        best = pso.run()

        assert best.fitness < 0.1, f"PSO słabo zbiegło dla dim={dim}. Fit: {best.fitness}"


class TestDE:
    def test_initialization(self, common_params):
        pop_size = 10
        de = DifferentialEvolution(pop_size=pop_size, **common_params)
        de.initialize()

        assert len(de.population) == pop_size

    def test_convergence(self, common_params):
        de = DifferentialEvolution(pop_size=30, F=0.5, CR=0.7, **common_params)
        de.max_iter = 80
        best = de.run()

        assert best.fitness < 0.1, f"DE nie zbiegło. Fitness: {best.fitness}"

    def test_boundary_constraints(self, common_params):
        de = DifferentialEvolution(pop_size=20, F=2.0, **common_params)
        de.run()

        for ind in de.population:
            assert np.all(ind.genom >= common_params['low'])
            assert np.all(ind.genom <= common_params['high'])

# population history tests

class TestHistory:

    @pytest.mark.parametrize("algo_class, extra_params", [
        (GA, {"pop_size": 10, "mutation_prob": 0.1}),
        (PSO, {"pop_size": 10}),
        (DifferentialEvolution, {"pop_size": 10}),
        (ES, {"mu": 5, "lam": 10})
    ])
    def test_history_length(self, algo_class, extra_params, common_params):

        params = common_params.copy()
        params.update(extra_params)
        runner = algo_class(**params)

        runner.run()

        expected_len = common_params['max_iter'] + 1
        actual_len = len(runner.history)

        assert actual_len == expected_len, \
            f"Error w {algo_class.__name__}: " \
            f"Historia ma {actual_len} wpisów, a wymagane jest {expected_len}. " \

    @pytest.mark.parametrize("algo_class, extra_params", [
        (GA, {"pop_size": 10}),
        (ES, {"mu": 5, "lam": 10}),
        (PSO, {"pop_size": 10}),
        (DifferentialEvolution, {"pop_size": 10})
    ])
    def test_population_history_length(self, algo_class, extra_params, common_params):

        params = common_params.copy()
        params.update(extra_params)
        runner = algo_class(**params)
        runner.run()

        expected_len = common_params['max_iter'] + 1
        actual_len = len(runner.population_history)

        assert actual_len == expected_len, \
            f"Error w {algo_class.__name__} (Population History): " \
            f"Zapisano {actual_len} klatek, a powinno być {expected_len}. " \
            f"Animacja będzie ucięta na starcie!"