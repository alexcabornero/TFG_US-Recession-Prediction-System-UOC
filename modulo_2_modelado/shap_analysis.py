"""
Análisis de explicabilidad SHAP del modelo ganador del Hito 2 (Tarea 2.9).

Calcula los SHAP values del Baseline Logistic Regression con
`class_weight='balanced'` (`models/final_model.pkl`) sobre el tramo
pre-hold-out (1967-01-31 → 2011-04-30, target_12m) — los mismos datos con
los que se entrenó el modelo. Para un modelo lineal, `shap.LinearExplainer`
es el explainer correcto: exacto, rápido y con interpretación matemática
directa de los coeficientes escalados.

Genera tres tipos de visualización en `docs/figures/`:
- Summary bar: importancia global como media del valor absoluto SHAP.
- Beeswarm: distribución de SHAP values por feature (signo + magnitud).
- Waterfall: tres casos representativos para narrativa de la memoria
  (verdadero positivo, falso negativo, verdadero negativo).
"""

import os
import warnings

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import shap

from baseline import cargar_dataset
from walk_forward_config import HOLDOUT_START


RUTA_MODELO = "models/final_model.pkl"
DIRECTORIO_FIGURAS = "docs/figures"
UMBRAL_MEDIANO = 0.654


def _seleccionar_indice(mascara: np.ndarray, scores: np.ndarray,
                        criterio: str) -> int | None:
    """Devuelve la posición del registro más representativo dentro de la máscara.

    Args:
        mascara: Booleana del mismo tamaño que `scores`.
        scores: Probabilidades predichas para cada observación.
        criterio: 'max' para elegir la observación con mayor probabilidad
            dentro de la máscara, 'min' para la menor.

    Returns:
        Índice posicional dentro del array completo, o None si la máscara
        está vacía.
    """
    indices = np.where(mascara)[0]
    if indices.size == 0:
        return None
    sub_scores = scores[indices]
    return int(indices[np.argmax(sub_scores) if criterio == 'max'
                       else np.argmin(sub_scores)])


def _guardar_figura(ruta: str) -> None:
    """Guarda la figura activa en disco con tight layout y cierra."""
    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()


def _exportar_waterfall(shap_values: shap.Explanation, idx: int,
                        ruta: str, titulo: str) -> None:
    """Genera y guarda un waterfall plot para una observación concreta."""
    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[idx], show=False, max_display=16)
    fig = plt.gcf()
    fig.suptitle(titulo, fontsize=11, fontweight='bold', y=1.02)
    _guardar_figura(ruta)


def main() -> None:
    pipeline = joblib.load(RUTA_MODELO)
    X, _, y_12m = cargar_dataset()

    mascara_pre_holdout = X.index < HOLDOUT_START
    X_train = X[mascara_pre_holdout]
    y_train = y_12m[mascara_pre_holdout]

    print("=" * 64)
    print("ANÁLISIS SHAP — MODELO GANADOR (Baseline LR balanced)")
    print("=" * 64)
    print(f"Modelo cargado desde: {RUTA_MODELO}")
    print(f"Datos pre-hold-out: {len(X_train)} obs "
          f"({X_train.index[0].date()} → {X_train.index[-1].date()})")
    print(f"Target: target_12m  |  Positivos: {int(y_train.sum())} "
          f"({y_train.mean():.1%})")
    print(f"Hold-out excluido: ≥ {HOLDOUT_START.date()}")

    scaler = pipeline.named_steps['scaler']
    classifier = pipeline.named_steps['classifier']

    X_scaled = scaler.transform(X_train)
    X_scaled_df = pd.DataFrame(
        X_scaled, columns=X_train.columns, index=X_train.index
    )

    explainer = shap.LinearExplainer(classifier, X_scaled_df)
    shap_values = explainer(X_scaled_df)

    importancia = np.abs(shap_values.values).mean(axis=0)
    ranking = pd.Series(importancia, index=X_train.columns).sort_values(
        ascending=False
    )

    print("\nRanking de features por importancia SHAP media absoluta")
    print("-" * 64)
    for posicion, (feature, valor) in enumerate(ranking.items(), start=1):
        print(f"{posicion:>2}. {feature:<25} {valor:.4f}")

    os.makedirs(DIRECTORIO_FIGURAS, exist_ok=True)

    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, show=False, max_display=16)
    _guardar_figura(os.path.join(DIRECTORIO_FIGURAS, "shap_summary_bar.png"))

    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, show=False, max_display=16)
    _guardar_figura(os.path.join(DIRECTORIO_FIGURAS, "shap_summary_beeswarm.png"))

    y_pred_proba = pipeline.predict_proba(X_train)[:, 1]
    y_pred = (y_pred_proba >= UMBRAL_MEDIANO).astype(int)
    y_real = y_train.values

    mascara_vp = (y_real == 1) & (y_pred == 1)
    mascara_fn = (y_real == 1) & (y_pred == 0)
    mascara_vn = (y_real == 0) & (y_pred == 0)

    idx_vp = _seleccionar_indice(mascara_vp, y_pred_proba, criterio='max')
    idx_fn = _seleccionar_indice(mascara_fn, y_pred_proba, criterio='max')
    idx_vn = _seleccionar_indice(mascara_vn, y_pred_proba, criterio='min')

    print("\nCasos representativos seleccionados (umbral mediano "
          f"= {UMBRAL_MEDIANO}):")
    casos = [
        ('VP (recesión correctamente detectada)', idx_vp,
         "shap_waterfall_vp.png"),
        ('FN (recesión NO detectada)', idx_fn,
         "shap_waterfall_fn.png"),
        ('VN (expansión correctamente clasificada)', idx_vn,
         "shap_waterfall_vn.png"),
    ]

    for etiqueta, idx, fichero in casos:
        if idx is None:
            print(f"  - {etiqueta}: sin observaciones disponibles")
            continue
        fecha = X_train.index[idx].date()
        proba = y_pred_proba[idx]
        real = int(y_real[idx])
        pred = int(y_pred[idx])
        print(f"  - {etiqueta}: fecha={fecha}  proba={proba:.3f}  "
              f"real={real}  pred={pred}")

        titulo = (f"{etiqueta} — {fecha}  |  "
                  f"P(recesión)={proba:.3f}  real={real}  pred={pred}")
        _exportar_waterfall(shap_values, idx,
                            os.path.join(DIRECTORIO_FIGURAS, fichero),
                            titulo)

    print("\nFiguras generadas en docs/figures/:")
    print("  - shap_summary_bar.png")
    print("  - shap_summary_beeswarm.png")
    print("  - shap_waterfall_vp.png")
    print("  - shap_waterfall_fn.png")
    print("  - shap_waterfall_vn.png")


if __name__ == "__main__":
    main()
