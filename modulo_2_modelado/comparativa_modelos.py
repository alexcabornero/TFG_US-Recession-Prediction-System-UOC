"""
Comparativa final de los cuatro modelos del Hito 2 — Tarea 2.7.

Lee los JSON de ablación generados por cada modelo (`baseline.py`,
`random_forest.py`, `xgboost_model.py`, `lightgbm_model.py`), construye una
tabla unificada de 4 modelos × 3 condiciones de rebalanceo y la formatea
ordenada por PR-AUC descendente, marcando con `*` el mejor valor de cada
columna. No entrena ningún modelo: solo agrega resultados existentes.

Artefactos generados:
- `models/comparativa_final.json`: datos completos para target_6m y target_12m
- `docs/comparativa_modelos_12m.csv`: tabla target_12m exportada
- `docs/comparativa_modelos_6m.csv`: tabla target_6m exportada
"""

import csv
import json
import os
import warnings

warnings.filterwarnings('ignore')

from typing import Dict, List

from walk_forward_config import CONDICIONES_REBALANCEO


MODELOS: Dict[str, str] = {
    'Baseline LR':   'models/ablacion_rebalanceo.json',
    'Random Forest': 'models/random_forest_ablacion.json',
    'XGBoost':       'models/xgboost_ablacion.json',
    'LightGBM':      'models/lightgbm_ablacion.json',
}

COLUMNAS_METRICA = ['pr_auc', 'auc_roc', 'recall', 'f1_score']


def cargar_resultados() -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """Carga los cuatro JSON de ablación en un diccionario unificado."""
    resultados: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for nombre_modelo, ruta in MODELOS.items():
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"No se encuentra {ruta}")
        with open(ruta, 'r') as f:
            resultados[nombre_modelo] = json.load(f)
    return resultados


def construir_filas(
    resultados: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    target: str,
) -> List[Dict]:
    """Convierte el diccionario de resultados en una lista plana de filas."""
    filas: List[Dict] = []
    for nombre_modelo in MODELOS:
        bloque_target = resultados[nombre_modelo].get(target, {})
        for cond in CONDICIONES_REBALANCEO:
            metricas = bloque_target.get(cond, {})
            if not metricas:
                continue
            filas.append({
                'modelo': nombre_modelo,
                'condicion': cond,
                'pr_auc': float(metricas['pr_auc']),
                'auc_roc': float(metricas['auc_roc']),
                'recall': float(metricas['recall']),
                'f1_score': float(metricas['f1_score']),
                'umbral_mediano': float(metricas.get('umbral_mediano', 0.0)),
            })
    filas.sort(key=lambda r: r['pr_auc'], reverse=True)
    return filas


def imprimir_tabla(filas: List[Dict], titulo: str) -> None:
    """Imprime la tabla formateada con `*` en el mejor valor de cada métrica."""
    if not filas:
        print(f"\n{titulo}\n(sin datos)")
        return

    mejores = {col: max(r[col] for r in filas) for col in COLUMNAS_METRICA}

    print("=" * 64)
    print(titulo)
    print("=" * 64)
    cabecera = (
        f"{'Modelo':<14} {'Condición':<10} "
        f"{'PR-AUC':>8} {'AUC-ROC':>9} {'Recall':>8} {'F1':>8}"
    )
    print(cabecera)
    for r in filas:
        celdas = []
        for col in COLUMNAS_METRICA:
            marca = '*' if r[col] == mejores[col] else ' '
            celdas.append(f"{r[col]:.3f}{marca}")
        print(
            f"{r['modelo']:<14} {r['condicion']:<10} "
            f"{celdas[0]:>8} {celdas[1]:>9} {celdas[2]:>8} {celdas[3]:>8}"
        )


def exportar_csv(filas: List[Dict], ruta: str) -> None:
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    columnas = ['modelo', 'condicion', 'pr_auc', 'auc_roc',
                'recall', 'f1_score', 'umbral_mediano']
    with open(ruta, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columnas)
        writer.writeheader()
        for fila in filas:
            writer.writerow({c: fila[c] for c in columnas})


if __name__ == "__main__":
    resultados = cargar_resultados()

    filas_12m = construir_filas(resultados, 'target_12m')
    filas_6m = construir_filas(resultados, 'target_6m')

    imprimir_tabla(filas_12m, "COMPARATIVA FINAL — TARGET 12M (principal)")
    print()
    imprimir_tabla(filas_6m, "COMPARATIVA FINAL — TARGET 6M (control)")

    os.makedirs("models", exist_ok=True)
    with open("models/comparativa_final.json", 'w', encoding='utf-8') as f:
        json.dump(
            {'target_12m': filas_12m, 'target_6m': filas_6m},
            f, indent=2, ensure_ascii=False,
        )

    exportar_csv(filas_12m, "docs/comparativa_modelos_12m.csv")
    exportar_csv(filas_6m, "docs/comparativa_modelos_6m.csv")