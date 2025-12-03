# app.py
import streamlit as st
import numpy as np
import pandas as pd

# Importowanie z poprawnych ścieżek pakietów:
# Upewnij się, że te importy są poprawne w stosunku do Twojej struktury folderów
from algorithms.EvolutionStrategy import ES 
from base.TestFunctions import Sphere, Rastrigin, Griewank, Rosenbrock 
from base.BaseAlgorithm import Individual # Nadal potrzebujemy Individual do history

# Mapowanie nazw funkcji na instancje klas callable
FUNCTIONS = {
    "Sphere": Sphere(),
    "Rastrigin": Rastrigin(),
    "Griewank": Griewank(),
    "Rosenbrock": Rosenbrock()
}

st.set_page_config(layout="centered", page_title="Prosty ES w Streamlit")

st.title("🔬 Prosty Optymalizator ES")
st.markdown("Uruchom algorytm Evolution Strategy (ES) i zobacz najlepsze wyniki z każdej generacji.")

# --- SIDEBAR: Ustawienia Algorytmu i Problemu ---
with st.sidebar:
    st.header("Konfiguracja Algorytmu")

    # Wybór funkcji
    selected_func_name = st.selectbox("Wybierz funkcję celu", list(FUNCTIONS.keys()))
    func = FUNCTIONS[selected_func_name]

    # Parametry Problemowe
    dim = st.slider("Wymiar (D)", 1, 10, 2) # Zmniejszyłem D dla prostoty
    low = st.number_input("Dolna granica (Low)", value=-5.0)
    high = st.number_input("Górna granica (High)", value=5.0)

    st.subheader("Parametry ES")
    mu = st.slider("μ (Liczba rodziców)", 2, 10, 5) # Uproszczone wartości
    lam = st.slider("λ (Liczba dzieci)", 5, 50, 20)
    max_iter = st.number_input("Maks. Liczba Generacji", 10, 500, 100) # Uproszczone wartości

# --- GŁÓWNA LOGIKA URUCHOMIENIOWA ---
if st.button("▶️ Uruchom Optymalizację"):
    st.subheader(f"Wyniki dla: **{selected_func_name}**")
    
    # 1. Inicjalizacja Algorytmu
    es_runner = ES(
        mu=mu, lam=lam, max_iter=max_iter, 
        func=func, dim=dim, low=low, high=high
    )
    
    # 2. Uruchomienie z Paskiem Postępu
    progress_bar = st.progress(0, text="Rozpoczynam optymalizację...")
    
    # Zwraca najlepszy obiekt Individual
    final_best_individual = es_runner.run_with_progress(progress_bar) 
    
    # 3. Wyświetlanie Końcowych Wyników
    st.success("Optymalizacja zakończona!")
    
    st.write("---")
    st.subheader("Najlepsze Rozwiązanie Globalne:")
    st.metric(label="Fitness", value=f"{final_best_individual.fitness:.6e}")
    st.code(f"Genom (x): {final_best_individual.genom}", language='python')
    st.write("---")

    # 4. Wykres Konwergencji (Historia Fitness)
    st.subheader("Historia Konwergencji (Najlepszy Fitness w Generacji)")
    
    # Przekształcanie listy obiektów Individual na DataFrame
    # Tworzymy listę słowników dla DataFrame
    history_data_for_df = [
        {"Generacja": i, "Najlepszy Fitness": ind.fitness}
        for i, ind in enumerate(es_runner.history)
    ]
    history_df = pd.DataFrame(history_data_for_df)
    
    st.line_chart(history_df.set_index('Generacja'))

    # 5. Wyświetlanie Najlepszych Obiektów z Iteracji
    st.subheader("Najlepsze Obiekty z Każdej Generacji")
    st.write("Tabela przedstawia najlepszy obiekt (genom i fitness) z każdej iteracji algorytmu.")

    # Tworzymy listę słowników dla DataFrame, pokazującą szczegóły każdego obiektu
    detailed_history_data = []
    for i, ind_obj in enumerate(es_runner.history):
        detailed_history_data.append({
            "Generacja": i,
            "Fitness": f"{ind_obj.fitness:.6e}",
            "Genom (x)": str(ind_obj.genom), # Konwertujemy NumPy array na string
            "Sigma": f"{ind_obj.sigma:.4f}" if ind_obj.sigma is not None else "N/A"
        })
    
    # Tworzymy DataFrame i wyświetlamy go
    detailed_history_df = pd.DataFrame(detailed_history_data)
    
    # Streamlit może wyświetlić duże tabele, ale dla bardzo wielu iteracji może być to nieefektywne
    st.dataframe(detailed_history_df)