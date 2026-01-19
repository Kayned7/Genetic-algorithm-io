import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from base.TestFunctions import EvalOnGrid
import plotly.express as px

from utils.config import FUNCTIONS
import pandas as pd


def plot_population_history(runner, low, high, max_iter, alg_name, func_name):
    if not hasattr(runner, 'population_history') or not runner.population_history:
        return None

    scatter_data = []
    step = 1 if max_iter <= 50 else max_iter // 50

    for gen_idx, pop_data in enumerate(runner.population_history):
        if gen_idx % step == 0:
            for i, ind_data in enumerate(pop_data):
                x_val = ind_data['genom'][0]
                y_val = ind_data['genom'][1]
                fit_val = ind_data['fitness']

                scatter_data.append({
                    "Generacja": gen_idx,
                    "ID Cząstki": i,
                    "x": x_val,
                    "y": y_val,
                    "Fitness": fit_val
                })

    if not scatter_data:
        return None

    df_anim = pd.DataFrame(scatter_data)

    # Marginesy
    margin_x = (high - low) * 0.1
    range_x = [low - margin_x, high + margin_x]
    range_y = [low - margin_x, high + margin_x]

    fig_anim = px.scatter(
        df_anim,
        x="x",
        y="y",
        animation_frame="Generacja",
        animation_group="ID Cząstki",
        color="Fitness",
        range_x=range_x,
        range_y=range_y,
        hover_name="ID Cząstki",
        title=f"Ewolucja - {alg_name}",
        color_continuous_scale="Viridis_r"
    )

    # Ustawienia animacji
    fig_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 100
    fig_anim.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
    fig_anim.update_layout(
        xaxis_title="x1",
        yaxis_title="x2",
        margin=dict(l=20, r=20, t=40, b=20),
        height=450  # Stała wysokość dla porządku w Arenie
    )

    # Marker celu
    fig_anim.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers",
            marker=dict(symbol="x", color="red", size=15),
            name="Cel",
            showlegend=False
        )
    )

    return fig_anim


def plot_2d(func, low, high, best_individual, title, dim):
    x = np.linspace(low, high, 200)
    y = np.linspace(low, high, 200)
    X, Y = np.meshgrid(x, y)
    Z = EvalOnGrid.eval_on_grid(func, X, Y)

    fig, ax = plt.subplots(figsize=(4, 3.5))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    cs = ax.contourf(X, Y, Z, levels=50, cmap="viridis")

    ax.tick_params(colors='white', which='both', labelsize=8)
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')

    cbar = fig.colorbar(cs, shrink=0.8)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white', fontsize=8)

    ax.scatter(best_individual.genom[0], best_individual.genom[1], color="red", s=30, marker="o", edgecolors='black',
               linewidth=1.0, zorder=10)
    suffix = " (Rzut)" if dim > 2 else ""
    ax.set_title(f"{title}{suffix}", fontsize=10)
    ax.set_xlabel("x1", fontsize=9)
    ax.set_ylabel("x2", fontsize=9)
    plt.tight_layout()
    return fig


def plot_3d(func, low, high, best_individual, title):
    x_lin = np.linspace(low, high, 50)
    y_lin = np.linspace(low, high, 50)
    X, Y = np.meshgrid(x_lin, y_lin)
    Z = EvalOnGrid.eval_on_grid(func, X, Y)

    fig = go.Figure()
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale="Viridis", showscale=True, opacity=0.9))
    fig.add_trace(go.Scatter3d(x=[best_individual.genom[0]], y=[best_individual.genom[1]], z=[best_individual.genom[2]],
                               mode="markers", marker=dict(size=5, color="red", symbol="circle")))
    fig.update_layout(title=title, scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="x3"),
                      margin=dict(l=0, r=0, b=0, t=40), height=400)
    return fig
