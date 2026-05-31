"""Página Acerca de — información del proyecto, autor, licencia y disclaimer."""

import streamlit as st


def renderizar():
    """Renderiza la página Acerca de."""
    st.title("ℹ️ Acerca del proyecto")
    st.divider()

    st.subheader("Sobre el TFG")
    st.markdown(
        """
        **Sistema de predicción de crisis financieras en EE.UU.** desarrollado
        como **Trabajo Final de Grado** del Grado en Ciencia de Datos
        Aplicada de la **Universitat Oberta de Catalunya (UOC)**.

        El proyecto aborda la pregunta: *¿es posible anticipar recesiones
        económicas con 12 meses de antelación usando exclusivamente
        indicadores macroeconómicos y financieros públicos?* Para
        responderla, se diseña un pipeline end-to-end que:

        1. **Adquiere datos** de FRED API y Yahoo Finance (1967-2025).
        2. **Construye un dataset maestro** de 16 indicadores macro-financieros
           alineados a frecuencia mensual.
        3. **Entrena y compara 4 familias de modelos** (Regresión Logística,
           Random Forest, XGBoost, LightGBM) bajo 3 condiciones de rebalanceo
           (`none`, `class_weight='balanced'`, SMOTE) en Walk-Forward CV.
        4. **Selecciona el modelo ganador por PR-AUC** y lo evalúa sobre un
           hold-out intocable de 165 observaciones (2011-2025).
        5. **Explica las predicciones con SHAP** y analiza la robustez frente
           al shock COVID.

        El resultado principal: un modelo lineal sencillo con **especificidad
        operativa perfecta en expansión** (0 falsos positivos en 9 años) y
        **AUC-ROC de 0.909**, cuyas decisiones se sustentan en indicadores
        económicamente coherentes (OECD CLI, desempleo, yield spread, credit
        spread, Treasury Bill).
        """
    )

    st.divider()

    st.subheader("Autor")
    st.markdown(
        """
        **Alejandro Cabornero López**
        Grado en Ciencia de Datos Aplicada · UOC
        📂 Repositorio: [github.com/AlexCaborneroLopez/TFG](https://github.com/alexcabornero/TFG_US-Recession-Prediction-System-UOC)
        """
    )

    st.divider()

    st.subheader("⚠️ Aviso legal y disclaimer académico")
    st.warning(
        """
        Esta herramienta tiene **fines exclusivamente académicos** y forma
        parte de un Trabajo Final de Grado universitario.

        **No constituye asesoramiento financiero ni recomendaciones de
        inversión.** Las predicciones del modelo se basan en datos
        históricos y patrones estadísticos; pueden fallar en la anticipación
        de shocks exógenos (pandemias, conflictos geopolíticos, crisis no
        previstas por indicadores macroeconómicos clásicos).

        El uso de esta información para la toma de decisiones financieras
        reales es **responsabilidad exclusiva del usuario**. Los autores no
        se hacen responsables de las consecuencias derivadas de tal uso.
        """
    )

    st.divider()

    st.subheader("📚 Referencias bibliográficas principales")
    st.markdown(
        """
        - **Bauer, M. D. & Mertens, T. M. (2018).** *Information in the Yield
          Curve about Future Recessions*. FRBSF Economic Letter.
        - **Bergmeir, C. & Benítez, J. M. (2012).** *On the use of
          cross-validation for time series predictor evaluation*. Information
          Sciences.
        - **Bernanke, B. & Blinder, A. (1992).** *The Federal Funds Rate and
          the Channels of Monetary Transmission*. American Economic Review.
        - **Estrella, A. & Mishkin, F. S. (1998).** *Predicting U.S.
          Recessions: Financial Variables as Leading Indicators*. Review of
          Economics and Statistics.
        - **Estrella, A. & Trubin, M. (2006).** *The Yield Curve as a Leading
          Indicator: Some Practical Issues*. FRBNY Current Issues in
          Economics and Finance.
        - **Gilchrist, S. & Zakrajšek, E. (2012).** *Credit Spreads and
          Business Cycle Fluctuations*. American Economic Review.
        - **OECD (2012).** *OECD System of Composite Leading Indicators*.
        - **Sahm, C. (2019).** *Direct Stimulus Payments to Individuals*.
          Hamilton Project / Brookings.
        - **Saito, T. & Rehmsmeier, M. (2015).** *The Precision-Recall Plot
          Is More Informative than the ROC Plot When Evaluating Binary
          Classifiers on Imbalanced Datasets*. PLOS ONE.
        - **Wright, J. H. (2006).** *The Yield Curve and Predicting
          Recessions*. Federal Reserve Board FEDS Working Paper 2006-07.
        """
    )

    st.divider()

    st.subheader("Stack tecnológico")
    st.markdown(
        """
        - **Lenguaje:** Python 3.10+
        - **Datos:** `fredapi`, `yfinance`, `pandas`, `numpy`
        - **Modelado:** `scikit-learn`, `xgboost`, `lightgbm`, `imbalanced-learn`
        - **Interpretabilidad:** `shap`
        - **Visualización:** `matplotlib`, `plotly`
        - **Aplicación:** `streamlit`
        - **Despliegue:** Streamlit Community Cloud
        - **Control de versiones:** Git / GitHub
        """
    )

    st.divider()

    st.caption(
        "📄 **Licencia:** Creative Commons BY-NC-ND 3.0 ES — © 2026 Alejandro "
        "Cabornero López. Se permite la reproducción y distribución no "
        "comercial con atribución, sin obras derivadas."
    )
