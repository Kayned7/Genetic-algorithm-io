# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from algorithms.EvolutionStrategy import ES
from algorithms.GeneticAlgorithm import GA
from base.TestFunctions import Sphere, Rastrigin, Griewank, Rosenbrock, Beale, BukinN6, EvalOnGrid
from base.BaseAlgorithm import Individual

FUNCTIONS = {
    "Sphere": Sphere(),
    "Rastrigin": Rastrigin(),
    "Griewank": Griewank(),
    "Rosenbrock": Rosenbrock(),
    "Beale": Beale(),
    "Bukin N.6": BukinN6()
}

st.set_page_config(layout="centered", page_title="Porównanie Algorytmów")

st.title("🔬 Platforma Badawcza Algorytmów")
st.markdown("Wybierz algorytm (ES lub GA), skonfiguruj parametry i porównaj wyniki.")

with st.sidebar:
    st.header("1. Wybierz Algorytm")
    alg_type = st.selectbox("Algorytm optymalizacyjny", ["Evolution Strategy (ES)", "Genetic Algorithm (GA)"])

    st.divider()
    st.header("2. Wybierz Funkcję Celu")
    selected_func_name = st.selectbox("Funkcja", list(FUNCTIONS.keys()))
    func = FUNCTIONS[selected_func_name]

    st.divider()
    st.header("3. Parametry Problemu")

    if selected_func_name in ["Beale", "Bukin N.6"]:
        st.warning(f"Funkcja {selected_func_name} jest zdefiniowana tylko dla 2 wymiarów.")
        dim = 2
    else:
        dim = st.slider("Wymiar (D)", 1, 20, 2)

    default_low = -5.0
    default_high = 5.0

    if selected_func_name == "Bukin N.6":
        default_low = -15.0
        default_high = 5.0
        st.info("Dla funkcji Bukin N.6 zalecany zakres zmiennych to [-15, 5].")

    col1, col2 = st.columns(2)
    with col1:
        low = st.number_input("Min (Low)", value=default_low)
    with col2:
        high = st.number_input("Max (High)", value=default_high)

    max_iter = st.number_input("Liczba Generacji", 10, 1000, 100)

    st.divider()
    st.header("4. Parametry Algorytmu")

    params = {}

    if alg_type == "Evolution Strategy (ES)":
        st.info("Ustawienia dla ES")
        params['mu'] = st.slider("μ (Rodzice)", 2, 50, 5)
        params['lam'] = st.slider("λ (Potomstwo)", 5, 100, 20)

    elif alg_type == "Genetic Algorithm (GA)":
        st.info("Ustawienia dla GA")
        params['pop_size'] = st.slider("Rozmiar populacji", 10, 200, 80)
        params['mutation_prob'] = st.slider("Prawdopodobieństwo mutacji", 0.0, 1.0, 0.1)
        params['crossover_prob'] = st.slider("Prawdopodobieństwo krzyżowania", 0.0, 1.0, 0.8)
        params['tournament_size'] = st.slider("Rozmiar turnieju", 2, 10, 3)

if st.button("▶️ Uruchom Optymalizację", type="primary"):
    st.subheader(f"Wyniki: {alg_type} na funkcji {selected_func_name}")

    runner = None

    if alg_type == "Evolution Strategy (ES)":
        runner = ES(
            func=func, dim=dim, low=low, high=high, max_iter=max_iter,
            mu=params['mu'], lam=params['lam']
        )
    elif alg_type == "Genetic Algorithm (GA)":
        runner = GA(
            func=func, dim=dim, low=low, high=high, max_iter=max_iter,
            pop_size=params['pop_size'],
            mutation_prob=params['mutation_prob'],
            crossover_prob=params['crossover_prob'],
            tournament_size=params['tournament_size']
        )

    progress_bar = st.progress(0, text="Inicjalizacja...")

    final_best_individual = runner.run_with_progress(progress_bar)

    st.success("Optymalizacja zakończona!")

    st.write("---")
    col_res1, col_res2 = st.columns([1, 2])

    with col_res1:
        st.metric(label="Najlepszy Fitness (min)", value=f"{final_best_individual.fitness:.6e}")

    with col_res2:
        st.caption("Znalezione współrzędne (Genom):")
        st.code(str(np.round(final_best_individual.genom, 4)), language='python')

    st.subheader("📉 Historia Konwergencji")

    history_data_for_df = [
        {"Generacja": i, "Najlepszy Fitness": ind.fitness}
        for i, ind in enumerate(runner.history)
    ]
    history_df = pd.DataFrame(history_data_for_df)

    st.line_chart(history_df.set_index('Generacja'))

    with st.expander("📊 Zobacz szczegółową tabelę wyników"):
        st.write("Tabela przedstawia najlepszy obiekt z każdej iteracji algorytmu.")

        detailed_history_data = []
        for i, ind_obj in enumerate(runner.history):
            row = {
                "Generacja": i,
                "Fitness": f"{ind_obj.fitness:.6e}",
                "Genom (x)": str(np.round(ind_obj.genom, 4)),
            }
            if ind_obj.sigma is not None:
                row["Sigma"] = f"{ind_obj.sigma:.4f}"
            else:
                row["Sigma"] = "-"

            detailed_history_data.append(row)

        detailed_history_df = pd.DataFrame(detailed_history_data)
        st.dataframe(detailed_history_df, width='stretch')

    x = np.linspace(low, high, 300)
    y = np.linspace(low, high, 300)
    X, Y = np.meshgrid(x, y)

    Z = EvalOnGrid.eval_on_grid(func, X, Y)

    fig, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contourf(X, Y, Z, levels=50, cmap="viridis")
    fig.colorbar(cs)
    ax.scatter(
        final_best_individual.genom[0],
        final_best_individual.genom[1],
        color="red",
        s=2,
        marker=".",
        label="Najlepsze rozwiązanie"
    )
    ax.set_title("Wykres funkcji")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    st.pyplot(fig)

    if dim > 2:
        fig = go.Figure()

        fig.add_trace(
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale="Viridis",
                showscale=True,
                opacity=0.9
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=[final_best_individual.genom[0]],
                y=[final_best_individual.genom[1]],
                z=[final_best_individual.genom[2]],
                mode="markers",
                marker=dict(
                    size=2,
                    color="red",
                    symbol="circle"
                ),
                name="Najlepsze rozwiązanie"
            )
        )

        fig.update_layout(
            title="Wykres funkcji",
            scene=dict(
                xaxis_title="x1",
                yaxis_title="x2",
                zaxis_title="f(x)"
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        st.plotly_chart(fig, use_container_width=True)