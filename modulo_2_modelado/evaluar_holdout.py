"""
Evaluación final del hold-out (Tarea 2.8).

Carga el modelo ganador serializado (`models/final_model.pkl`) y lo evalúa
sobre el hold-out intocable (≥ 2011-05-31), segmentado en sub-ventanas:
- Expansión pura (may 2011 → feb 2020): mide tasa de falsas alarmas.
- COVID (mar-abr 2020): caso de estudio separado.
- Global: solo como referencia, no debe promediarse sin segmentar.

Persiste los resultados en `models/holdout_resultados.json` y los imprime
por pantalla. Solo debe ejecutarse UNA VEZ tras la confirmación del tutor
sobre el modelo ganador.
"""

import json
import os
import sys

import joblib

sys.path.insert(0, os.path.dirname(__file__))

from baseline import BaselineModel, cargar_dataset


RUTA_MODELO = "models/final_model.pkl"
RUTA_ABLACION = "models/ablacion_rebalanceo.json"
RUTA_SALIDA = "models/holdout_resultados.json"


def cargar_umbral_ganador() -> float:
    """Lee el umbral mediano OOF del modelo ganador (LR balanced, target_12m).

    Returns:
        Umbral de decisión mediano calculado durante el walk-forward CV.
    """
    with open(RUTA_ABLACION, "r", encoding="utf-8") as f:
        ablacion = json.load(f)
    return float(ablacion["target_12m"]["balanced"]["umbral_mediano"])


def main():
    print("=" * 70)
    print("EVALUACIÓN HOLD-OUT — MODELO GANADOR (Baseline LR balanced)")
    print("=" * 70)

    pipeline = joblib.load(RUTA_MODELO)
    umbral = cargar_umbral_ganador()
    print(f"Modelo cargado desde: {RUTA_MODELO}")
    print(f"Umbral mediano OOF: {umbral:.6f}")

    X, _, y_12m = cargar_dataset()
    print(f"Dataset completo: {len(X)} obs "
          f"({X.index[0].date()} → {X.index[-1].date()})")

    modelo = BaselineModel(rebalanceo="balanced")
    modelo.modelo = pipeline
    modelo.umbral_optimo = umbral

    resultado = modelo.evaluar_hold_out(X, y_12m)

    os.makedirs(os.path.dirname(RUTA_SALIDA), exist_ok=True)
    with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

    print(f"\nResultados persistidos en: {RUTA_SALIDA}")


if __name__ == "__main__":
    main()
