"""Página Predicción — serie temporal de probabilidad de recesión + gauge."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from componentes.carga_datos import (
    calcular_probabilidades,
    cargar_modelo,
    separar_features_y_targets,
)
from componentes.estilos import (
    COLOR_NBER,
    COLOR_NARANJA,
    COLOR_PROBABILIDAD,
    COLOR_UMBRAL,
    COLOR_VERDE,
    OPACIDAD_BANDAS_NBER,
    UMBRAL_DECISION,
    estado_probabilidad,
)


def _calcular_periodos_recesion(usrec: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Convierte la serie binaria USREC en intervalos contiguos de recesión.

    Args:
        usrec: Serie binaria con 1 en los meses de recesión NBER.

    Returns:
        Lista de tuplas (inicio, fin) para cada periodo continuo de recesión.
    """
    periodos = []
    en_recesion = False
    inicio = None
    for fecha, valor in usrec.items():
        if valor == 1 and not en_recesion:
            inicio = fecha
            en_recesion = True
        elif valor == 0 and en_recesion:
            periodos.append((inicio, fecha_anterior))
            en_recesion = False
        fecha_anterior = fecha
    if en_recesion:
        periodos.append((inicio, fecha_anterior))
    return periodos


def _construir_grafico(
    proba: pd.Series,
    usrec: pd.Series,
) -> go.Figure:
    """Construye el gráfico Plotly de probabilidad vs tiempo con bandas NBER."""
    fig = go.Figure()

    for inicio, fin in _calcular_periodos_recesion(usrec):
        fig.add_vrect(
            x0=inicio, x1=fin,
            fillcolor=COLOR_NBER, opacity=OPACIDAD_BANDAS_NBER,
            layer="below", line_width=0,
        )

    fig.add_trace(go.Scatter(
        x=proba.index, y=proba.values,
        mode="lines",
        name="P(recesión)",
        line=dict(color=COLOR_PROBABILIDAD, width=1.8),
        hovertemplate="<b>%{x|%Y-%m}</b><br>P(recesión)=%{y:.3f}<extra></extra>",
    ))

    fig.add_hline(
        y=UMBRAL_DECISION, line_dash="dash",
        line_color=COLOR_UMBRAL, line_width=1.5,
        annotation_text=f"Umbral {UMBRAL_DECISION:.3f}",
        annotation_position="top right",
        annotation_font=dict(color=COLOR_UMBRAL),
    )

    fig.update_layout(
        xaxis_title="Fecha",
        yaxis_title="P(recesión a 12 meses)",
        yaxis=dict(range=[-0.02, 1.02]),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=10, b=10),
        height=340,
        plot_bgcolor="white",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EAEAEA")
    fig.update_yaxes(showgrid=True, gridcolor="#EAEAEA")

    return fig


def _construir_gauge(proba_actual: float) -> go.Figure:
    """Construye el gauge Plotly con la probabilidad actual y umbrales."""
    etiqueta, color, _ = estado_probabilidad(proba_actual)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba_actual * 100,
        number={"suffix": " %", "font": {"size": 36}},
        title={"text": etiqueta, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.30},
            "steps": [
                {"range": [0, 30], "color": "#E8F5E9"},
                {"range": [30, UMBRAL_DECISION * 100], "color": "#FFF4E6"},
                {"range": [UMBRAL_DECISION * 100, 100], "color": "#FCE8E8"},
            ],
            "threshold": {
                "line": {"color": COLOR_UMBRAL, "width": 3},
                "thickness": 0.85,
                "value": UMBRAL_DECISION * 100,
            },
        },
    ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=20, b=10),
        height=200,
    )
    return fig


def renderizar():
    """Renderiza la página Predicción."""
    st.title("📈 Probabilidad de recesión a 12 meses")
    st.caption("Serie temporal del modelo final sobre 1967 → 2025.")
    st.divider()

    pipeline = cargar_modelo()
    X, usrec, _, _ = separar_features_y_targets()
    proba = calcular_probabilidades(pipeline, X)

    proba_actual = float(proba.iloc[-1])
    fecha_actual = proba.index[-1]
    etiqueta, color, emoji = estado_probabilidad(proba_actual)
    distancia = (proba_actual - UMBRAL_DECISION) * 100

    col_gauge, col_kpis = st.columns([1, 2])

    with col_gauge:
        st.plotly_chart(
            _construir_gauge(proba_actual),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with col_kpis:
        st.subheader(f"{emoji} Estado actual: {etiqueta}")
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric(
            "P(recesión)",
            f"{proba_actual * 100:.1f} %",
        )
        kpi2.metric(
            "Umbral de alerta",
            f"{UMBRAL_DECISION * 100:.1f} %",
        )
        kpi3.metric(
            "Distancia al umbral",
            f"{distancia:+.1f} pp",
        )
        st.caption(
            f"Última observación disponible: **{fecha_actual.strftime('%Y-%m-%d')}**"
        )

    st.plotly_chart(
        _construir_grafico(proba, usrec),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.info(
        "💡 **Cómo leer este gráfico.** La línea azul es la probabilidad de "
        "recesión a 12 meses estimada por el modelo en cada fecha. La línea "
        "roja discontinua marca el umbral de decisión calibrado en el "
        "walk-forward CV. Las bandas grises señalan los periodos de recesión "
        "reales según NBER (variable `USREC = 1`). El modelo emite alerta "
        "cuando la línea azul cruza el umbral."
    )
