"""Página Overview — bienvenida y KPIs principales."""

import pandas as pd
import streamlit as st

from componentes.carga_datos import (
    calcular_probabilidades,
    cargar_modelo,
    separar_features_y_targets,
)
from componentes.estilos import UMBRAL_DECISION, estado_probabilidad


def _ultima_recesion_nber(usrec: pd.Series) -> str:
    """Devuelve la última recesión NBER como rango legible 'YYYY-MM → YYYY-MM'."""
    indices_uno = usrec[usrec == 1].index
    if len(indices_uno) == 0:
        return "Sin recesiones en el periodo"

    fin = indices_uno[-1]
    inicio = fin
    for i in range(len(indices_uno) - 2, -1, -1):
        delta_meses = (indices_uno[i + 1] - indices_uno[i]).days
        if delta_meses > 45:
            break
        inicio = indices_uno[i]
    return f"{inicio.strftime('%Y-%m')} → {fin.strftime('%Y-%m')}"


def renderizar():
    """Renderiza la página Overview."""
    st.title("📊 Predicción de Crisis Financieras en EE.UU.")
    st.caption("TFG · Grado en Ciencia de Datos Aplicada · UOC")
    st.divider()

    st.markdown(
        """
        Sistema end-to-end de **predicción de recesiones económicas** en
        Estados Unidos basado en indicadores macro-financieros del periodo
        1967-2025. Combina **16 indicadores** de FRED y Yahoo Finance con un
        modelo de **Regresión Logística** validado mediante **Walk-Forward
        Cross-Validation** y explicado con **valores SHAP**.

        **Modelo final:** Regresión Logística con `class_weight='balanced'`
        sobre `target_12m` (horizonte de 12 meses). Seleccionado por PR-AUC
        entre 12 combinaciones de modelo × rebalanceo (Baseline LR, Random
        Forest, XGBoost, LightGBM × `none`, `balanced`, SMOTE).
        """
    )

    st.divider()

    pipeline = cargar_modelo()
    X, usrec, _, _ = separar_features_y_targets()
    proba = calcular_probabilidades(pipeline, X)

    proba_actual = float(proba.iloc[-1])
    fecha_actual = proba.index[-1]
    etiqueta, _, emoji = estado_probabilidad(proba_actual)
    ultima_recesion = _ultima_recesion_nber(usrec)

    st.subheader("Estado actual del riesgo de recesión")
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "P(recesión) a 12 meses",
        f"{proba_actual * 100:.1f} %",
        delta=f"Umbral {UMBRAL_DECISION * 100:.1f} %",
        delta_color="off",
    )
    col2.metric(
        "Estado",
        f"{emoji} {etiqueta}",
    )
    col3.metric(
        "Última observación",
        fecha_actual.strftime("%Y-%m-%d"),
    )

    st.caption(f"📅 Última recesión NBER registrada: **{ultima_recesion}**")

    st.divider()

    st.subheader("Explora la aplicación")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown("**📚 Variables**")
        st.caption("Diccionario de los 16 indicadores agrupados por categoría económica.")
    with c2:
        st.markdown("**📈 Predicción**")
        st.caption("Serie temporal de probabilidades 1967-2025 con bandas NBER y umbral.")
    with c3:
        st.markdown("**🔍 SHAP**")
        st.caption("Importancia global de indicadores y casos representativos VP / FN / VN.")
    with c4:
        st.markdown("**🧪 Backtesting**")
        st.caption("Walk-Forward CV, hold-out segmentado y comparativa de modelos.")
    with c5:
        st.markdown("**ℹ️ Acerca de**")
        st.caption("Proyecto, autor, referencias bibliográficas y licencia.")

    st.info(
        "💡 Usa el menú de navegación en la barra lateral izquierda para "
        "explorar cada sección."
    )
