import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go


from utils.config import CSS_STYLES, ALG_NAMES
from utils.ui import render_problem_sidebar, render_algo_params, reset_comparison_state
from utils.plots import  plot_2d, plot_3d, plot_population_history
from utils.factory import get_runner


# --- CONFIG ---
st.set_page_config(layout="wide", page_title="Platforma Optymalizacji")
st.markdown(CSS_STYLES, unsafe_allow_html=True)

# PAGE 1: POJEDYNCZA ANALIZA
def view_single_analysis():
    reset_comparison_state()
    st.title("🔬 Platforma Badawcza Algorytmów")
    st.markdown("Wybierz algorytm (ES, GA,DE lub PSO), skonfiguruj parametry i przeanalizuj wyniki.")

    func, func_name, dim, low, high, max_iter = render_problem_sidebar()


    st.sidebar.divider()
    st.sidebar.header("⚙️ Algorytm")

    alg_type = st.sidebar.selectbox("Wybierz Algorytm", ALG_NAMES)
    st.sidebar.write(f"**Parametry: {alg_type}**")

    with st.sidebar:
        params = render_algo_params(alg_type, "single_sidebar")

    st.markdown("---")

    if st.button("▶️ Uruchom Optymalizację", type="primary", use_container_width=True):
        runner = get_runner(alg_type, func, dim, low, high, max_iter, params)
        progress_bar = st.progress(0, text="Inicjalizacja...")
        final_best = runner.run_with_progress(progress_bar)
        st.success("Gotowe!")

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Najlepszy Fitness", f"{final_best.fitness:.6e}")
        with c2:
            st.code(str(np.round(final_best.genom, 4)), language='python')

        tab_titles = ["📉 Zbieżność", "🗺️ Wyniki 2D"]
        if dim > 2:
            tab_titles.append("🌌 Wyniki 3D")
        tab_titles.extend(["🎬 Dynamika Populacji", "📊 Dane"])

        tabs = st.tabs(tab_titles)

        # 1. Zbieżność
        with tabs[0]:
            hist_data = [{"Generacja": i, "Fitness": ind.fitness} for i, ind in enumerate(runner.history)]
            st.plotly_chart(px.line(hist_data, x="Generacja", y="Fitness", log_y=True), use_container_width=True)

        # 2. 2D
        with tabs[1]:
            col_spacer1, col_plot, col_spacer2 = st.columns([1, 2, 1])
            with col_plot:
                st.pyplot(plot_2d(func, low, high, final_best, f"Wykres funkcji (rzut 2D) - {alg_type}", dim),
                          use_container_width=True)

        current_tab_idx = 2

        # 3. 3D
        if dim > 2:
            with tabs[current_tab_idx]:
                st.plotly_chart(plot_3d(func, low, high, final_best, f"Wykres przestrzeni rozwiązań (x1, x2, x3) - {alg_type}"),
                                use_container_width=True)
            current_tab_idx += 1

        # 4. Animacja
        with tabs[current_tab_idx]:
            st.subheader("🎬 Dynamika Populacji")

            fig_anim = plot_population_history(runner, low, high, max_iter, alg_type, func_name)
            if fig_anim:
                st.plotly_chart(fig_anim, use_container_width=True)
            else:
                st.warning("Błąd generowania animacji (brak historii populacji).")

        # 5. Dane
        with tabs[current_tab_idx + 1]:
            st.dataframe(pd.DataFrame([{"Gen": i, "Fit": f"{x.fitness:.6e}"} for i, x in enumerate(runner.history)]),
                         use_container_width=True)



# PAGE 2: PORÓWNANIE

def view_comparison():
    func, func_name, dim, low, high, max_iter = render_problem_sidebar(on_change_callback=reset_comparison_state)

    st.title("⚔️ Porównanie Algorytmów")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Algorytm A")
        a1 = st.selectbox("Wybierz A", ALG_NAMES, key="a1", on_change=reset_comparison_state)
        with st.expander("Parametry A", expanded=True): p1 = render_algo_params(a1, "A")
    with c2:
        st.subheader("Algorytm B")
        a2 = st.selectbox("Wybierz B", ALG_NAMES, index=1, key="a2", on_change=reset_comparison_state)
        with st.expander("Parametry B", expanded=True): p2 = render_algo_params(a2, "B")

    st.divider()

    if st.button("▶️ Rozpocznij porównanie", type="primary", use_container_width=True):
        r1 = get_runner(a1, func, dim, low, high, max_iter, p1)
        r2 = get_runner(a2, func, dim, low, high, max_iter, p2)

        col1, col2 = st.columns(2)
        with col1: b1 = r1.run_with_progress(st.progress(0, text=f"{a1}..."))
        with col2: b2 = r2.run_with_progress(st.progress(0, text=f"{a2}..."))

        st.session_state['arena_results'] = {'r1': r1, 'r2': r2, 'b1': b1, 'b2': b2, 'a1': a1, 'a2': a2}

    if 'arena_results' in st.session_state:
        res = st.session_state['arena_results']
        r1, r2, b1, b2 = res['r1'], res['r2'], res['b1'], res['b2']

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.metric(res['a1'], f"{b1.fitness:.4e}")
        with c2:
            better = b2.fitness < b1.fitness
            st.metric(res['a2'], f"{b2.fitness:.4e}", delta="Lepszy" if better else "Gorszy", delta_color="inverse")

        # Dynamiczne zakładki z animacją w środku
        tab_titles = ["📉 Zbieżność", "🗺️ Wykres 2D"]
        if dim > 2:
            tab_titles.append("🌌Wykres 3D")
        tab_titles.extend(["🎬 Animacja", "📊 Dane"])

        tabs = st.tabs(tab_titles)

        # Zbieżność
        with tabs[0]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=[i.fitness for i in r1.history], name=res['a1'], line=dict(color='blue')))
            fig.add_trace(go.Scatter(y=[i.fitness for i in r2.history], name=res['a2'], line=dict(color='red')))
            fig.update_layout(yaxis_type="log", title="Porównanie Fitness")
            st.plotly_chart(fig, use_container_width=True)

        #2D
        with tabs[1]:
            cc1, cc2 = st.columns(2)
            with cc1: st.pyplot(plot_2d(func, low, high, b1, res['a1'], dim), use_container_width=True)
            with cc2: st.pyplot(plot_2d(func, low, high, b2, res['a2'], dim), use_container_width=True)

        current_tab_idx = 2

        # 3D
        if dim > 2:
            with tabs[current_tab_idx]:
                cc1, cc2 = st.columns(2)
                with cc1:
                    st.plotly_chart(plot_3d(func, low, high, b1, res['a1']), use_container_width=True)
                with cc2:
                    st.plotly_chart(plot_3d(func, low, high, b2, res['a2']), use_container_width=True)
            current_tab_idx += 1

        # Animacja
        with tabs[current_tab_idx]:
            st.subheader("🎬 Dynamika Populacji")
            ac1, ac2 = st.columns(2)

            with ac1:
                fig_anim1 = plot_population_history(r1, low, high, max_iter, res['a1'], func_name)
                if fig_anim1:
                    st.plotly_chart(fig_anim1, use_container_width=True)
                else:
                    st.write("Brak danych animacji dla A.")

            with ac2:
                fig_anim2 = plot_population_history(r2, low, high, max_iter, res['a2'], func_name)
                if fig_anim2:
                    st.plotly_chart(fig_anim2, use_container_width=True)
                else:
                    st.write("Brak danych animacji dla B.")

        # Tab: Dane
        with tabs[current_tab_idx + 1]:
            dc1, dc2 = st.columns(2)
            with dc1:
                st.write(f"**Historia {res['a1']}**")
                df1 = pd.DataFrame([{"Gen": i, "Fit": f"{x.fitness:.6e}"} for i, x in enumerate(r1.history)])
                st.dataframe(df1, use_container_width=True, height=300)
            with dc2:
                st.write(f"**Historia {res['a2']}**")
                df2 = pd.DataFrame([{"Gen": i, "Fit": f"{x.fitness:.6e}"} for i, x in enumerate(r2.history)])
                st.dataframe(df2, use_container_width=True, height=300)

pg = st.navigation([
    st.Page(view_single_analysis, title="Pojedyncza Analiza", icon="👤"),
    st.Page(view_comparison, title="Porównanie", icon="⚔️"),
])

pg.run()