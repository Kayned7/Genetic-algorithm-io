from base.TestFunctions import Sphere, Rastrigin, Griewank, Rosenbrock, Beale, BukinN6

CSS_STYLES = """
<style>
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 5rem !important;
    }
    div[data-testid="stSidebarNav"] span {
        font-size: 1.25rem !important; 
        padding-top: 5px;
        padding-bottom: 5px;
    }
    div[data-testid="stSidebarNav"] {
        padding-top: 1rem !important;
    }
</style>
"""

# --- DATA & CONSTANTS ---
FUNCTIONS = {
    "Sphere": Sphere(),
    "Rastrigin": Rastrigin(),
    "Griewank": Griewank(),
    "Rosenbrock": Rosenbrock(),
    "Beale": Beale(),
    "Bukin N.6": BukinN6()
}

ALG_NAMES = [
    "Evolution Strategy (ES)",
    "Genetic Algorithm (GA)",
    "Particle Swarm Optimization (PSO)",
    "Differential Evolution (DE)"
]

