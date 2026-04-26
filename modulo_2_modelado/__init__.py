"""
Módulo de Modelado y Evaluación

Este módulo contiene los algoritmos de machine learning para la predicción
de recesiones económicas, incluyendo:
- Regresión Logística (baseline)
- Random Forest
- XGBoost
- LightGBM

Todos los modelos implementan validación temporal (Walk-Forward CV) y
métricas orientadas a minimizar falsos negativos (Recall prioritario).
"""

__version__ = "1.0.0"
