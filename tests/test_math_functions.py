import pytest
import numpy as np
from base.TestFunctions import Sphere, Rastrigin, Rosenbrock, Griewank, Beale, BukinN6, EvalOnGrid


# fixtures

@pytest.fixture
def sphere(): return Sphere()


@pytest.fixture
def rastrigin(): return Rastrigin()


@pytest.fixture
def rosenbrock(): return Rosenbrock()


@pytest.fixture
def griewank(): return Griewank()


@pytest.fixture
def beale(): return Beale()


@pytest.fixture
def bukin(): return BukinN6()


# global min tests

def test_sphere_minimum(sphere):
    x = np.zeros(5)
    assert np.isclose(sphere(x), 0.0), "Sphere: Błąd kalibracji w punkcie [0...0]"


def test_rastrigin_minimum(rastrigin):
    x = np.zeros(3)
    assert np.isclose(rastrigin(x), 0.0), "Rastrigin: Błąd kalibracji w punkcie [0...0]"


def test_rosenbrock_minimum(rosenbrock):
    x = np.ones(4)
    assert np.isclose(rosenbrock(x), 0.0), "Rosenbrock: Błąd kalibracji w punkcie [1...1]"


def test_griewank_minimum(griewank):
    x = np.zeros(10)
    assert np.isclose(griewank(x), 0.0, atol=1e-6), "Griewank: Błąd kalibracji w punkcie [0...0]"


def test_beale_minimum(beale):
    x = np.array([3.0, 0.5])
    assert np.isclose(beale(x), 0.0), "Beale: Błąd kalibracji w punkcie [3, 0.5]"


def test_bukin_n6_minimum(bukin):
    x = np.array([-10.0, 1.0])
    assert np.isclose(bukin(x), 0.0), "Bukin N.6: Błąd kalibracji w punkcie [-10, 1]"


# structural and integrity tests

def test_dimensionality_flexibility(sphere):
    assert np.isscalar(sphere(np.ones(1))), "Sphere nie obsługuje 1D"
    assert np.isscalar(sphere(np.ones(20))), "Sphere nie obsługuje 20D"


def test_rosenbrock_shape_handling(rosenbrock):
    val = rosenbrock(np.array([1.0, 2.0]))
    # (1-1)^2 + 100*(2-1^2)^2 = 0 + 100*1 = 100
    assert np.isclose(val, 100.0)


def test_eval_on_grid_structure():
    X = np.array([[0, 1], [0, 1]])
    Y = np.array([[0, 0], [1, 1]])

    sphere_func = Sphere()

    Z = EvalOnGrid.eval_on_grid(sphere_func, X, Y)

    assert Z.shape == X.shape
    assert isinstance(Z, np.ndarray)

    assert Z[0, 0] == 0.0

    assert Z[1, 1] == 2.0


# sanity checks

def test_inputs_as_lists(sphere):
    try:
        sphere([1, 2])
    except TypeError:
        # To jest akceptowalne, jeśli wymagamy numpy array
        pass
    except Exception as e:
        pytest.fail(f"Nieoczekiwany błąd przy podaniu listy: {e}")