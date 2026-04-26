"""
Serialización del modelo ganador final (Tarea 2.7.6).

Entrena el Baseline LR balanced sobre todos los datos pre-hold-out
(ene 1967 – abr 2011, 532 observaciones) y serializa el pipeline
en models/final_model.pkl. Ejecutar una única vez tras la selección
del modelo ganador — nunca reentrenar con datos del hold-out.
"""

import sys
import os
import joblib

sys.path.insert(0, os.path.dirname(__file__))

from baseline import BaselineModel, cargar_dataset
from walk_forward_config import HOLDOUT_START


def main():
    X, _, y_12m = cargar_dataset()

    mascara = X.index < HOLDOUT_START
    X_train = X[mascara]
    y_train = y_12m[mascara]

    print(f"Entrenando sobre {len(X_train)} obs "
          f"({X_train.index[0].date()} → {X_train.index[-1].date()})")

    modelo = BaselineModel(rebalanceo='balanced')
    modelo.entrenar_final(X_train, y_train)
    modelo.guardar_modelo('models/final_model.pkl')

    pipeline = joblib.load('models/final_model.pkl')
    clf = pipeline.named_steps['classifier']
    print(f"Verificado — class_weight: {clf.class_weight} | "
          f"coef shape: {clf.coef_.shape}")


if __name__ == "__main__":
    main()
