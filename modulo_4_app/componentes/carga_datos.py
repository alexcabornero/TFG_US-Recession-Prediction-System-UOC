"""Cargadores cacheados del dataset, modelo y artefactos del Hito 2.

Todas las rutas se resuelven respecto a la raíz del proyecto para que la app
funcione independientemente del CWD desde el que se lance Streamlit.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
RUTA_DATASET = RAIZ_PROYECTO / "data" / "processed" / "dataset_final.csv"
RUTA_MODELO = RAIZ_PROYECTO / "models" / "final_model.pkl"
RUTA_HOLDOUT = RAIZ_PROYECTO / "models" / "holdout_resultados.json"
RUTA_BASELINE_METRICS = RAIZ_PROYECTO / "models" / "baseline_metrics_12m.json"
RUTA_COVID = RAIZ_PROYECTO / "models" / "analisis_covid.json"
RUTA_COMPARATIVA = RAIZ_PROYECTO / "models" / "comparativa_final.json"
RUTA_FIGURAS = RAIZ_PROYECTO / "docs" / "figures"


@st.cache_data(show_spinner=False)
def cargar_dataset() -> pd.DataFrame:
    """Carga el dataset final maestro completo con índice temporal mensual."""
    df = pd.read_csv(RUTA_DATASET, index_col=0, parse_dates=True)
    df.index.name = "fecha"
    return df


@st.cache_data(show_spinner=False)
def separar_features_y_targets() -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Devuelve X (features), usrec, target_6m y target_12m alineados.

    Returns:
        Tupla (X, usrec, y_6m, y_12m) con el mismo índice temporal.
    """
    df = cargar_dataset()
    X = df.drop(columns=["target_6m", "target_12m", "usrec"])
    return X, df["usrec"], df["target_6m"], df["target_12m"]


@st.cache_resource(show_spinner=False)
def cargar_modelo():
    """Carga el pipeline serializado del modelo ganador."""
    return joblib.load(RUTA_MODELO)


@st.cache_data(show_spinner=False)
def cargar_holdout() -> dict:
    """Carga las métricas del hold-out segmentado."""
    with open(RUTA_HOLDOUT, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def cargar_baseline_metrics() -> dict:
    """Carga las métricas walk-forward del baseline LR balanced."""
    with open(RUTA_BASELINE_METRICS, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def cargar_analisis_covid() -> dict:
    """Carga el análisis de robustez COVID."""
    with open(RUTA_COVID, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def cargar_comparativa() -> dict:
    """Carga la comparativa final de las 12 combinaciones modelo × rebalanceo."""
    with open(RUTA_COMPARATIVA, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def calcular_probabilidades(_pipeline, X: pd.DataFrame) -> pd.Series:
    """Calcula las probabilidades de recesión para todo el dataset.

    El argumento `_pipeline` lleva guion bajo para que Streamlit no intente
    hashearlo (es un objeto sklearn no hashable).

    Args:
        _pipeline: Pipeline sklearn cargado con `cargar_modelo`.
        X: DataFrame con las features (mismas columnas que en entrenamiento).

    Returns:
        Serie con `predict_proba(X)[:, 1]` indexada por fecha.
    """
    proba = _pipeline.predict_proba(X)[:, 1]
    return pd.Series(proba, index=X.index, name="proba_recesion")
