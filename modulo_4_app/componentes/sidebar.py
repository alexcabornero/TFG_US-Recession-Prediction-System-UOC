"""Sidebar común a todas las páginas de la aplicación.

Contiene navegación entre páginas y disclaimer académico fijo.
"""

import streamlit as st

PAGINAS = [
    "Overview",
    "Variables",
    "Predicción",
    "SHAP",
    "Backtesting",
    "Acerca de",
]


def renderizar_sidebar() -> str:
    """Renderiza el sidebar y devuelve la página seleccionada por el usuario."""
    with st.sidebar:
        st.markdown("## 📊 Predicción de Crisis Financieras")
        st.caption("TFG · Grado en Ciencia de Datos Aplicada · UOC")
        st.divider()

        pagina = st.radio(
            "Navegación",
            PAGINAS,
            label_visibility="visible",
        )

        st.divider()

        st.markdown("### ⚠️ Disclaimer")
        st.caption(
            "Esta herramienta tiene fines exclusivamente académicos y "
            "no constituye asesoramiento financiero ni recomendaciones "
            "de inversión."
        )

        st.divider()
        st.caption(
            "📄 CC BY-NC-ND 3.0 ES · © 2026 Alejandro Cabornero López"
        )

    return pagina
