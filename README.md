# 🧬 Platforma Badawcza Algorytmów / Algorithm Research Platform

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-3F4F75)

[English version below]

## 🇵🇱 O Projekcie
**Platforma Badawcza Algorytmów** to interaktywne narzędzie webowe służące do wizualizacji, analizy i porównywania działania algorytmów optymalizacji heurystycznej (bezgradientowej). Aplikacja pozwala użytkownikom na głębokie zrozumienie dynamiki populacji oraz zbieżności różnych metod optymalizacji w czasie rzeczywistym.

Narzędzie umożliwia przeprowadzanie pojedynczych symulacji oraz bezpośrednie porównywanie skuteczności dwóch algorytmów (Head-to-Head) na wybranych funkcjach testowych (np. Sphere, Rastrigin).

### ✨ Główne funkcjonalności
* **Algorytmy:** Genetic Algorithm (GA), Evolution Strategy (ES), Particle Swarm Optimization (PSO), Differential Evolution (DE).
* **Wizualizacja:** Wykresy zbieżności (Fitness), rzuty 2D z mapami ciepła, wizualizacja przestrzenna 3D.
* **Porównanie:** Tryb "Benchmark" zestawiający dwa algorytmy obok siebie.
* **Konfiguracja:** Pełna parametryzacja (rozmiar populacji, mutacja, inercja, $\mu, \lambda$).

### 📷 Galeria

#### 1. Konfiguracja i Pojedyncza Analiza
<img width="1840" height="842" alt="Zrzut ekranu 2026-01-27 190846" src="https://github.com/user-attachments/assets/642d4799-f554-422e-97d9-9c6f2bc42609" />

*Panel główny aplikacji i wybór funkcji celu.*

#### 2. Wizualizacja Wyników (2D i 3D)
| Rzut 2D (Heatmapa) | Wizualizacja 3D |
|:---:|:---:|
| ![2D Plot]<img width="658" height="495" alt="Zrzut ekranu 2026-01-27 191006" src="https://github.com/user-attachments/assets/5aa5abd4-4585-4d66-94ef-4c3e17cdf7bd" />| ![3D Plot]<img width="1360" height="617" alt="Zrzut ekranu 2026-01-27 191254" src="https://github.com/user-attachments/assets/3fae3b26-0315-4da9-ad09-c83494c1d943" />|

#### 3. Dynamika i Zbieżność
<img width="1377" height="805" alt="Zrzut ekranu 2026-01-27 190953" src="https://github.com/user-attachments/assets/2a39a441-be2d-485d-84cf-c192b68c0eba" />
*Wykres zbieżności funkcji dopasowania (Fitness) w czasie.*

#### 4. Tryb Porównania (Benchmark)
<img width="1768" height="753" alt="Zrzut ekranu 2026-01-27 191130" src="https://github.com/user-attachments/assets/e4663282-58ff-4ac5-ae3d-097e3011f913" />
*Konfiguracja porównania dwóch algorytmów.*

<img width="1385" height="754" alt="Zrzut ekranu 2026-01-27 191158" src="https://github.com/user-attachments/assets/b6ebad96-490c-456a-a36b-e138c41abdca" />
*Porównanie szybkości zbieżności (np. ES vs GA).*

---

### 🚀 Instalacja i Uruchomienie

1.  **Wymagania:** Python 3.8+, zainstalowane biblioteki z `requirements.txt`.
    ```bash
    pip install streamlit numpy pandas plotly
    ```

2.  **Uruchomienie:**
    Aplikacja domyślnie korzysta z portu 5998. Uruchom ją poleceniem:
    ```bash
    streamlit run app.py --server.port 5998
    ```
    Możliwe jest także ustawienie własnego portu, wystarczy wtedy zmienić cztery ostatnie cyfry polecenia:
    ```bash
    streamlit run app.py --server.port ABCD
    ```
3.  Otwórz w przeglądarce (lub inny wybrany port): `http://localhost:5998`

### 📂 Struktura Projektu

```text
├── algorithms/                 # Implementacja algorytmów (GA, ES, PSO, DE)
│   ├── base_algorithm.py
│   ├── genetic_algorithm.py
│   ├── ...
├── functions/                  # Definicje funkcji testowych (Sphere, Rastrigin itp.)
├── visualization/              # Logika wykresów Plotly
├── app.py                      # Główny plik aplikacji Streamlit
└── requirements.txt            # Zależności
