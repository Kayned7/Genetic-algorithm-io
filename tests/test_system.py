import pytest
import unittest.mock as mock
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.config import ALG_NAMES, FUNCTIONS
from utils.factory import get_runner
from utils.plots import plot_population_history, plot_2d, plot_3d
from base.TestFunctions import Sphere


#fixtures

@pytest.fixture
def mock_streamlit():
    with mock.patch('utils.plots.st') as mock_st:
        yield mock_st


@pytest.fixture
def standard_params():
    return {
        "mu": 5, "lam": 20,  # ES
        "pop_size": 10,  # GA, DE, PSO
        "mutation_prob": 0.1,  # GA
        "crossover_prob": 0.8,  # GA
        "tournament_size": 3,  # GA
        "w": 0.5, "c1": 1.5, "c2": 1.5,  # PSO
        "F": 0.5, "CR": 0.7  # DE
    }


def get_valid_params(alg_name, all_params):
    if "Evolution Strategy" in alg_name:
        return {k: v for k, v in all_params.items() if k in ['mu', 'lam']}
    elif "Genetic Algorithm" in alg_name:
        return {k: v for k, v in all_params.items() if
                k in ['pop_size', 'mutation_prob', 'crossover_prob', 'tournament_size']}
    elif "Particle Swarm" in alg_name:
        return {k: v for k, v in all_params.items() if k in ['pop_size', 'w', 'c1', 'c2']}
    elif "Differential Evolution" in alg_name:
        return {k: v for k, v in all_params.items() if k in ['pop_size', 'F', 'CR']}
    return {}


#integrity tests

def test_factory_completeness(standard_params):
    func = Sphere()
    dim = 2

    for alg_name in ALG_NAMES:
        algo_params = get_valid_params(alg_name, standard_params)

        runner = get_runner(alg_name, func, dim, -5, 5, 10, algo_params)

        assert runner is not None, f"Fabryka zwróciła None dla {alg_name}"

        try:
            best = runner.run()
            assert best is not None
        except Exception as e:
            pytest.fail(f"BŁĄD WYKONANIA dla '{alg_name}': {str(e)}")


def test_factory_invalid_input(standard_params):
    runner = get_runner("Nieistniejący Algorytm", Sphere(), 2, -5, 5, 10, standard_params)
    assert runner is None

def test_plot_population_history_integration(standard_params, mock_streamlit):
    alg_name = "Genetic Algorithm (GA)"

    # POPRAWKA: Filtrujemy parametry
    algo_params = get_valid_params(alg_name, standard_params)

    runner = get_runner(alg_name, Sphere(), 2, -5, 5, 20, algo_params)
    runner.run()

    fig = plot_population_history(runner, -5, 5, 20, alg_name, "Sphere")

    assert fig is not None
    assert isinstance(fig, go.Figure)
    assert len(fig.frames) > 0


def test_plot_2d_integration(standard_params):
    alg_name = "Particle Swarm Optimization (PSO)"

    algo_params = get_valid_params(alg_name, standard_params)

    runner = get_runner(alg_name, Sphere(), 2, -5, 5, 10, algo_params)
    best = runner.run()

    fig = plot_2d(Sphere(), -5, 5, best, "Test 2D", 2)
    assert fig is not None
    assert len(fig.axes) > 0


# scenario tests

def test_scenario_switching_dimensions(standard_params):
    func = Sphere()
    alg_name = "Differential Evolution (DE)"

    algo_params = get_valid_params(alg_name, standard_params)

    runner_2d = get_runner(alg_name, func, 2, -5, 5, 10, algo_params)
    res_2d = runner_2d.run()
    assert len(res_2d.genom) == 2

    runner_10d = get_runner(alg_name, func, 10, -5, 5, 10, algo_params)
    res_10d = runner_10d.run()
    assert len(res_10d.genom) == 10

    last_pop = runner_10d.population_history[-1]
    assert len(last_pop[0]['genom']) == 10


def test_scenario_missing_history_handling(mock_streamlit):

    class EmptyRunner:
        population_history = []

    fig = plot_population_history(EmptyRunner(), -5, 5, 10, "Test", "Func")
    assert fig is None