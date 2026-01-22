import streamlit as st
from utils.config import FUNCTIONS

def render_algo_params(alg_type, key_suffix):
    params = {}
    if alg_type == "Evolution Strategy (ES)":
        params['mu'] = st.slider(f"μ (Rodzice)", 2, 50, 5, key=f"mu_{key_suffix}")
        params['lam'] = st.slider(f"λ (Potomstwo)", 5, 100, 20, key=f"lam_{key_suffix}")
    elif alg_type == "Genetic Algorithm (GA)":
        params['pop_size'] = st.slider(f"Rozmiar populacji", 10, 200, 50, key=f"pop_{key_suffix}")
        params['mutation_prob'] = st.slider(f"Prawdopodobieństwo mutacji", 0.0, 1.0, 0.1, key=f"mut_{key_suffix}")
        params['crossover_prob'] = st.slider(f"Prawdopodobieństwo krzyżowania", 0.0, 1.0, 0.8,
                                             key=f"cross_{key_suffix}")
        params['tournament_size'] = st.slider(f"Rozmiar turnieju", 2, 10, 3, key=f"tour_{key_suffix}")
    elif alg_type == "Particle Swarm Optimization (PSO)":
        params['pop_size'] = st.slider(f"Liczba cząstek", 10, 200, 30, key=f"pso_pop_{key_suffix}")
        params['w'] = st.slider(f"Inercja (w)", 0.0, 1.0, 0.5, key=f"w_{key_suffix}")
        params['c1'] = st.number_input(f"Kognitywny (c1)", 0.0, 4.0, 1.5, step=0.1, key=f"c1_{key_suffix}")
        params['c2'] = st.number_input(f"Socjalny (c2)", 0.0, 4.0, 1.5, step=0.1, key=f"c2_{key_suffix}")
    elif alg_type == "Differential Evolution (DE)":
        params['pop_size'] = st.slider(f"Rozmiar populacji", 10, 200, 50, key=f"de_pop_{key_suffix}")
        params['F'] = st.slider(f"Mutacja (F)", 0.0, 2.0, 0.5, step=0.01, key=f"F_{key_suffix}")
        params['CR'] = st.slider(f"Krzyżowanie (CR)", 0.0, 1.0, 0.7, step=0.05, key=f"CR_{key_suffix}")
    return params

def render_problem_sidebar(on_change_callback=None):
    st.sidebar.header("🧮 Definicja Funkcji Testowej")

    selected_func_name = st.sidebar.selectbox(
        "Funkcja Celu",
        list(FUNCTIONS.keys()),
        on_change=on_change_callback
    )
    func = FUNCTIONS[selected_func_name]

    if selected_func_name in ["Beale", "Bukin N.6"]:
        st.sidebar.warning("Ta funkcja wymusza D=2.")
        dim = 2
    else:
        dim = st.sidebar.slider("Wymiarowość (D)", 2, 20, 2, on_change=on_change_callback)

    default_low, default_high = (-15.0, 5.0) if selected_func_name == "Bukin N.6" else (-5.0, 5.0)
    c1, c2 = st.sidebar.columns(2)
    with c1:
        low = st.number_input("Min", value=default_low)
    with c2:
        high = st.number_input("Max", value=default_high)
    if low >= high:
        st.sidebar.warning("Odwrócone granice (Min > Max). Zostaną one automatycznie zamienione miejscami.")
    max_iter = st.sidebar.number_input("Liczba Generacji", 10, 2000, 100)

    return func, selected_func_name, dim, low, high, max_iter

def reset_comparison_state():
    if 'arena_results' in st.session_state:
        del st.session_state['arena_results']