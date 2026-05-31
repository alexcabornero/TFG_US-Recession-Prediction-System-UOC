"""Punto de entrada de la aplicación Streamlit del TFG.

Ejecutar desde la raíz del proyecto:
    streamlit run modulo_4_app/app.py

La app carga el modelo serializado del Hito 2 (`models/final_model.pkl`) y
el dataset congelado (`data/processed/dataset_final.csv`). No realiza
ningún reentrenamiento ni consulta APIs externas en runtime.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from componentes.sidebar import renderizar_sidebar
from paginas import (
    acerca_de,
    backtesting,
    overview,
    prediccion,
    shap_explicabilidad,
    variables,
)


st.set_page_config(
    page_title="Predicción de Crisis Financieras EE.UU.",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


PAGINAS = {
    "Overview": overview.renderizar,
    "Variables": variables.renderizar,
    "Predicción": prediccion.renderizar,
    "SHAP": shap_explicabilidad.renderizar,
    "Backtesting": backtesting.renderizar,
    "Acerca de": acerca_de.renderizar,
}


def main():
    pagina = renderizar_sidebar()
    PAGINAS[pagina]()


if __name__ == "__main__":
    main()
