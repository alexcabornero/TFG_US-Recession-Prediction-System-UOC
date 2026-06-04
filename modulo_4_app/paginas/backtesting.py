"""Página Backtesting — validación histórica del modelo.

Cuatro bloques organizados en tabs:
  1. Walk-Forward CV: métricas por fold sobre 1967-2011.
  2. Hold-Out segmentado: métricas en expansión, COVID y global (2011-2025).
  3. Caso de estudio COVID: análisis de la limitación estructural del
     benchmark NBER post-2008.
  4. Comparativa de modelos: ranking de las 12 combinaciones del Hito 2.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from componentes.carga_datos import (
    RUTA_FIGURAS,
    cargar_baseline_metrics,
    cargar_comparativa,
    cargar_holdout,
)

RUTA_COVID_PNG = RUTA_FIGURAS / "anticipacion_covid.png"


def _tab_walk_forward():
    st.subheader("Walk-Forward Cross-Validation")
    st.markdown(
        "El modelo se validó con **4 folds expandientes** sobre el periodo "
        "**1967-2011**, respetando el orden temporal y con un gap entre "
        "train y test para prevenir data leakage."
    )

    datos = cargar_baseline_metrics()
    folds = datos["folds"]

    df = pd.DataFrame([
        {
            "Fold": f["fold"],
            "Train": f"{f['train_start'][:7]} → {f['train_end'][:7]}",
            "Test": f"{f['test_start'][:7]} → {f['test_end'][:7]}",
            "N test": f["test_size"],
            "Recesiones": f["n_recesiones_test"],
            "PR-AUC": f["pr_auc"],
            "AUC-ROC": f["auc_roc"],
            "Recall": f["recall"],
            "F1": f["f1_score"],
        }
        for f in folds
    ])

    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "PR-AUC": st.column_config.NumberColumn(format="%.3f"),
            "AUC-ROC": st.column_config.NumberColumn(format="%.3f"),
            "Recall": st.column_config.NumberColumn(format="%.3f"),
            "F1": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    agregado = datos.get("aggregated", {})
    if agregado:
        st.markdown("**Métricas agregadas:**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PR-AUC", f"{agregado.get('pr_auc_mean', 0):.3f}")
        c2.metric("AUC-ROC", f"{agregado.get('auc_roc_mean', 0):.3f}")
        c3.metric("Recall", f"{agregado.get('recall_mean', 0):.3f}")
        c4.metric("F1", f"{agregado.get('f1_mean', 0):.3f}")

    st.markdown(
        f"""
        **Umbral óptimo seleccionado:** `{datos['umbral_optimo']:.4f}` (mediana
        de los umbrales de de los 4 folds).

        💡 La variabilidad por fold es esperable: los primeros folds tienen
        menos datos de entrenamiento y los periodos de test pueden contener
        regímenes económicos distintos. La métrica agregada es la referencia
        global; los folds individuales informan sobre la estabilidad temporal.
        """
    )


def _tab_holdout():
    st.subheader("Hold-Out intocable (2011-2025)")
    st.markdown(
        "El hold-out se reservó tras la selección final del modelo. Las "
        "métricas se reportan **segmentadas por sub-ventana** porque los "
        "únicos 2 positivos `target_12m` corresponden a la anticipación del "
        "shock COVID, exógeno por naturaleza. Promediar expansión y COVID "
        "ocultaría los resultados reales."
    )

    datos = cargar_holdout()

    def _fila(sub: str, etiqueta: str):
        d = datos[sub]
        return {
            "Sub-ventana": etiqueta,
            "Rango": f"{d['fecha_inicio'][:7]} → {d['fecha_fin'][:7]}",
            "N": d["n_observaciones"],
            "Recesiones": d["n_recesiones_real"],
            "Predichas": d["n_recesiones_pred"],
            "Recall": d["recall"],
            "AUC-ROC": d["auc_roc"],
            "PR-AUC": d["pr_auc"],
            "False Alarm Rate": d["false_alarm_rate"],
        }

    df = pd.DataFrame([
        _fila("expansion", "Expansión"),
        _fila("covid", "COVID (shock)"),
        _fila("global", "Global (referencia)"),
    ])

    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Recall": st.column_config.NumberColumn(format="%.3f"),
            "AUC-ROC": st.column_config.NumberColumn(format="%.3f"),
            "PR-AUC": st.column_config.NumberColumn(format="%.3f"),
            "False Alarm Rate": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    st.success(
        "✅ **Resultado clave — Especificidad operativa perfecta en expansión.** "
        "Durante 9 años de fase expansiva (mayo 2011 → febrero 2020), el "
        "modelo genera **0 falsos positivos sobre 104 observaciones negativas**. "
        "Para un sistema de alerta temprana, esta propiedad es operativamente "
        "más importante que un Recall alto: las falsas alarmas erosionan la "
        "credibilidad del sistema."
    )

    st.info(
        "📊 **AUC-ROC = 0.909 en expansión** confirma que el modelo **ordena "
        "correctamente las observaciones según riesgo**. Las probabilidades "
        "absolutas no cruzan el umbral, pero su estructura relativa es "
        "informativa — la visualización en la página *Predicción* permite "
        "observar tendencias y picos relativos."
    )


def _tab_covid():
    st.subheader("Caso de estudio: anticipación del COVID")

    st.markdown(
        """
        Los únicos 2 positivos `target_12m` del hold-out son la anticipación
        del COVID (2019-03/04 apuntando a 2020-03/04). El modelo **no los
        detecta** y este resultado merece una explicación cuidadosa.
        """
    )

    _, columna_central, _ = st.columns([1, 2, 1])
    with columna_central:
        st.image(str(RUTA_COVID_PNG), use_container_width=True)

    st.markdown(
        """
        **El COVID es un shock exógeno** que rompe la cadena causal sobre la
        que se construye el modelo:

        1. La **curva de tipos** refleja expectativas del mercado sobre
           crecimiento futuro y política monetaria. El mercado **no preveía
           una pandemia** — no podía.
        2. Los **credit spreads** reflejan calidad crediticia esperada. Las
           empresas **no eran objetivamente más riesgosas en 2019** por
           causa COVID — el riesgo era invisible para el mercado.
        3. El **indicador líder OECD CLI** agrega señales económicas reales
           (manufactura, permisos, mercado). Ninguna de estas señales contiene
           información sobre pandemias.

        **Implicación metodológica para el TFG:** el benchmark de evaluación
        post-2008 está **estructuralmente sesgado**. La Gran Recesión 2008-09
        cae en el periodo pre-hold-out (entrenamiento); las únicas 2
        recesiones del hold-out son la COVID, exógena por naturaleza. El
        benchmark efectivo se reduce a *"¿anticipa el modelo un shock no
        contenido en sus features?"*, pregunta cuya respuesta es **no por
        construcción del problema**, no por fallo del modelo.

        Cualquier sistema basado exclusivamente en macroindicadores
        tradicionales habría producido el mismo resultado.
        """
    )


def _tab_comparativa():
    st.subheader("Comparativa de las 12 combinaciones del Hito 2")
    st.markdown(
        "Resultado del benchmark sistemático **4 modelos × 3 condiciones de "
        "rebalanceo** evaluados con Walk-Forward CV sobre `target_12m`. "
        "La métrica primaria es **PR-AUC** por el desbalance de clases "
        "(12,2 % de positivos)."
    )

    datos = cargar_comparativa()
    df = pd.DataFrame(datos["target_12m"]).sort_values("pr_auc", ascending=False)
    df["Combinación"] = df["modelo"] + " (" + df["condicion"] + ")"

    tabla = df[[
        "Combinación", "pr_auc", "auc_roc", "recall", "f1_score",
    ]].rename(columns={
        "pr_auc": "PR-AUC",
        "auc_roc": "AUC-ROC",
        "recall": "Recall",
        "f1_score": "F1",
    })

    st.dataframe(
        tabla,
        hide_index=True,
        use_container_width=True,
        column_config={
            "PR-AUC": st.column_config.NumberColumn(format="%.3f"),
            "AUC-ROC": st.column_config.NumberColumn(format="%.3f"),
            "Recall": st.column_config.NumberColumn(format="%.3f"),
            "F1": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    pr_auc_invertido = df["pr_auc"].iloc[::-1].tolist()
    combinacion_invertida = df["Combinación"].iloc[::-1].tolist()
    colores = ["#94A3B8"] * len(df)
    colores[0] = "#1E3A8A"
    colores_invertidos = colores[::-1]

    fig = go.Figure(go.Bar(
        x=pr_auc_invertido,
        y=combinacion_invertida,
        orientation="h",
        marker_color=colores_invertidos,
        text=[f"{v:.3f}" for v in pr_auc_invertido],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>PR-AUC=%{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="PR-AUC",
        yaxis_title=None,
        margin=dict(l=10, r=40, t=20, b=10),
        height=420,
        plot_bgcolor="white",
        showlegend=False,
    )
    fig.update_xaxes(
        range=[0, max(df["pr_auc"]) * 1.15],
        showgrid=True, gridcolor="#EAEAEA",
    )
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    ganador = df.iloc[0]
    st.success(
        f"🏆 **Modelo ganador: {ganador['modelo']} ({ganador['condicion']})** "
        f"con PR-AUC = {ganador['pr_auc']:.3f}, Recall = {ganador['recall']:.3f}. "
        "Supera a todos los modelos no lineales pese a su simplicidad: en un "
        "dataset pequeño y desbalanceado con relaciones macroeconómicas "
        "fundamentalmente monótonas, la regresión logística regularizada "
        "explota óptimamente la señal disponible."
    )

    st.caption(
        "💡 PR-AUC (Precision-Recall AUC) es la métrica de referencia en "
        "problemas desbalanceados: penaliza tanto los falsos positivos como "
        "los falsos negativos sin verse inflada por la clase mayoritaria. "
        "Un clasificador aleatorio obtendría PR-AUC ≈ 0,122 (la prevalencia)."
    )


def renderizar():
    """Renderiza la página Backtesting."""
    st.title("🧪 Validación histórica del modelo")
    st.caption("Walk-Forward CV (1967-2011) + Hold-Out intocable (2011-2025).")
    st.divider()

    tab_wf, tab_ho, tab_covid, tab_comp = st.tabs([
        "Walk-Forward",
        "Hold-Out",
        "Caso COVID",
        "Comparativa de modelos",
    ])

    with tab_wf:
        _tab_walk_forward()

    with tab_ho:
        _tab_holdout()

    with tab_covid:
        _tab_covid()

    with tab_comp:
        _tab_comparativa()
