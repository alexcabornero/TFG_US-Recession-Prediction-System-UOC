"""Página SHAP — explicabilidad del modelo final.

Sirve las figuras pre-generadas en la Tarea 2.9 del Hito 2 (no calcula SHAP
en runtime para mantener la app ligera y desplegable en Streamlit Community
Cloud). Cada figura va acompañada de un texto interpretativo anclado en la
literatura económica relevante.
"""

import streamlit as st

from componentes.carga_datos import RUTA_FIGURAS

RUTA_SUMMARY_BAR = RUTA_FIGURAS / "shap_summary_bar.png"
RUTA_BEESWARM = RUTA_FIGURAS / "shap_summary_beeswarm.png"
RUTA_WATERFALL_VP = RUTA_FIGURAS / "shap_waterfall_vp.png"
RUTA_WATERFALL_FN = RUTA_FIGURAS / "shap_waterfall_fn.png"
RUTA_WATERFALL_VN = RUTA_FIGURAS / "shap_waterfall_vn.png"


def _layout_grafico_texto(ruta_imagen, descripcion_markdown: str) -> None:
    """Renderiza una imagen a la izquierda (40 %) y su descripción a la derecha (60 %)."""
    col_imagen, col_texto = st.columns([2, 3])
    with col_imagen:
        st.image(str(ruta_imagen), use_container_width=True)
    with col_texto:
        st.markdown(descripcion_markdown)


def _tab_importancia():
    st.subheader("Importancia global de los indicadores")
    _layout_grafico_texto(
        RUTA_SUMMARY_BAR,
        """
        El gráfico muestra la magnitud media de los valores SHAP por feature,
        es decir, **cuánto contribuye cada indicador, en promedio, a la
        decisión del modelo**.

        - **`oecd_cli_us`** (Composite Leading Indicator de la OECD) lidera el
          ranking. Es un compuesto multivariante que agrega señales adelantadas
          de manufactura, permisos de construcción y mercado, validado oficialmente
          como predictor de *turning points* (OECD, 2012).
        - **`unrate`** (tasa de desempleo) ocupa el segundo lugar, coherente con
          la *Sahm Rule* (Sahm, 2019), que detecta recesiones a partir de
          aumentos sostenidos del desempleo.
        - **`yield_spread` (T10Y3M)** y **`credit_spread` (BAA-AAA)** aparecen
          como predictores complementarios. El yield spread ha precedido todas
          las recesiones de EE.UU. desde 1969 (Estrella & Mishkin, 1998); el
          credit spread refleja deterioro de calidad crediticia (Gilchrist &
          Zakrajšek, 2012).
        - **`tb3ms`** (Treasury Bill a 3 meses) cierra el top-5, capturando
          fases tardías del ciclo de subidas de tipos previas a recesión.
        """,
    )


def _tab_distribucion():
    st.subheader("Distribución de impactos individuales (beeswarm)")
    _layout_grafico_texto(
        RUTA_BEESWARM,
        """
        Cada punto representa una observación; su **color** indica el valor
        relativo del indicador (rojo = alto, azul = bajo) y su **posición
        horizontal** indica la dirección del impacto SHAP sobre la predicción.

        Permite ver **cómo cada indicador empuja la probabilidad de recesión**:
        valores altos de yield spread (rojo a la izquierda) empujan hacia
        expansión; valores bajos o negativos (azul a la derecha) empujan hacia
        recesión. Coherente con la teoría de la curva de tipos invertida.
        """,
    )


def _tab_caso(titulo: str, ruta_imagen, descripcion: str):
    st.subheader(titulo)
    _layout_grafico_texto(ruta_imagen, descripcion)


def renderizar():
    """Renderiza la página SHAP."""
    st.title("🔍 ¿Qué indicadores impulsan las predicciones?")
    st.caption("Análisis SHAP sobre 532 obs pre-hold-out (1967-2011).")
    st.divider()

    st.markdown(
        "Los gráficos de esta página explican **qué indicadores macroeconómicos "
        "influyen más en la decisión del modelo** y cómo se combinan en casos "
        "individuales representativos. El cálculo SHAP se realizó con "
        "`shap.LinearExplainer` sobre el modelo final, excluyendo las 165 "
        "observaciones del hold-out para evitar contaminación."
    )

    tab_imp, tab_dist, tab_vp, tab_fn, tab_vn = st.tabs([
        "Importancia",
        "Distribución",
        "Caso VP",
        "Caso FN",
        "Caso VN",
    ])

    with tab_imp:
        _tab_importancia()

    with tab_dist:
        _tab_distribucion()

    with tab_vp:
        _tab_caso(
            "Caso VP — Verdadero Positivo",
            RUTA_WATERFALL_VP,
            """
            Observación de recesión real correctamente clasificada por el
            modelo (probabilidad ≥ umbral). El gráfico waterfall descompone
            la predicción mostrando **qué indicadores empujaron la
            probabilidad por encima del umbral**. Los valores en rojo
            incrementan la P(recesión); los azules la reducen.

            Permite validar que las decisiones positivas del modelo están
            **respaldadas por señales económicas coherentes** (yield spread
            negativo, credit spread elevado, indicador líder en descenso).
            """,
        )

    with tab_fn:
        _tab_caso(
            "Caso FN — Falso Negativo",
            RUTA_WATERFALL_FN,
            """
            Observación de recesión real **no detectada** por el modelo
            (probabilidad < umbral). El waterfall muestra qué indicadores
            "compensaron" las señales positivas, manteniendo la probabilidad
            por debajo del umbral.

            Esta visualización es valiosa para **diagnóstico de errores**:
            identificar combinaciones de features que confunden al modelo
            y orientar futuras mejoras en feature engineering.
            """,
        )

    with tab_vn:
        _tab_caso(
            "Caso VN — Verdadero Negativo",
            RUTA_WATERFALL_VN,
            """
            Observación de expansión correctamente clasificada con baja
            probabilidad de recesión. La mayoría de features empujan hacia
            la izquierda (rojo a azul invertido = expansión), confirmando
            que el modelo lee correctamente las señales de estabilidad.
            """,
        )

    st.divider()
    st.caption(
        "📚 **Referencias:** OECD (2012); Sahm (2019); Estrella & Mishkin (1998); "
        "Gilchrist & Zakrajšek (2012); Wright (2006). Ver página *Acerca de* "
        "para la lista completa."
    )
