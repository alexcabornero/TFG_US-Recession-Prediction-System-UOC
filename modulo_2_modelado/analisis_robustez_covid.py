"""
Análisis de Robustez COVID (Tarea 2.10).

Cuantifica el comportamiento del modelo ganador en la ventana de
anticipación del COVID: para cada mes t entre 2018-01 y 2020-02,
calcula P(recesión) y la compara con el umbral de decisión (0.6543).
Los target_12m positivos en esta ventana apuntan a meses de la recesión
NBER del COVID (feb-abr 2020). El objetivo NO es validar el modelo
contra una recesión clásica, sino documentar por qué un shock exógeno
no es anticipable con indicadores macro a 12 meses.
"""

import json
import os
import sys

import joblib
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(__file__))

from baseline import cargar_dataset


RUTA_MODELO = "models/final_model.pkl"
RUTA_ABLACION = "models/ablacion_rebalanceo.json"
RUTA_SALIDA_JSON = "models/analisis_covid.json"
RUTA_SALIDA_FIG = "docs/figures/anticipacion_covid.png"

VENTANA_INICIO = pd.Timestamp("2018-01-31")
VENTANA_FIN = pd.Timestamp("2020-02-29")


def cargar_umbral_ganador() -> float:
    with open(RUTA_ABLACION, "r", encoding="utf-8") as f:
        ablacion = json.load(f)
    return float(ablacion["target_12m"]["balanced"]["umbral_mediano"])


def construir_tabla(
    X: pd.DataFrame,
    y_12m: pd.Series,
    pipeline,
    umbral: float,
) -> pd.DataFrame:
    """Construye la tabla mes a mes con proba, target real, predicción y distancia al umbral."""
    mascara = (X.index >= VENTANA_INICIO) & (X.index <= VENTANA_FIN)
    X_v = X[mascara]
    y_v = y_12m[mascara]

    proba = pipeline.predict_proba(X_v)[:, 1]
    pred = (proba >= umbral).astype(int)

    tabla = pd.DataFrame({
        "fecha": X_v.index,
        "target_12m_real": y_v.values.astype(int),
        "proba_recesion": proba.round(4),
        "umbral": umbral,
        "prediccion": pred,
        "distancia_umbral": (proba - umbral).round(4),
    })
    return tabla


def resumen_estadistico(tabla: pd.DataFrame, umbral: float) -> dict:
    """Sintetiza el comportamiento del modelo en la ventana."""
    positivos = tabla[tabla["target_12m_real"] == 1]
    negativos = tabla[tabla["target_12m_real"] == 0]

    return {
        "ventana_inicio": str(VENTANA_INICIO.date()),
        "ventana_fin": str(VENTANA_FIN.date()),
        "n_observaciones": len(tabla),
        "umbral": umbral,
        "positivos_target_12m": {
            "n": int(len(positivos)),
            "fechas": [str(f.date()) for f in positivos["fecha"]],
            "proba_media": float(positivos["proba_recesion"].mean())
                if len(positivos) > 0 else None,
            "proba_max": float(positivos["proba_recesion"].max())
                if len(positivos) > 0 else None,
            "proba_min": float(positivos["proba_recesion"].min())
                if len(positivos) > 0 else None,
            "detectados": int(positivos["prediccion"].sum()),
            "no_detectados": int(len(positivos) - positivos["prediccion"].sum()),
        },
        "negativos_target_12m": {
            "n": int(len(negativos)),
            "proba_media": float(negativos["proba_recesion"].mean()),
            "proba_max": float(negativos["proba_recesion"].max()),
            "falsos_positivos": int(negativos["prediccion"].sum()),
        },
        "max_proba_global_ventana": float(tabla["proba_recesion"].max()),
        "fecha_max_proba": str(
            tabla.loc[tabla["proba_recesion"].idxmax(), "fecha"].date()
        ),
    }


def generar_grafico(tabla: pd.DataFrame, umbral: float) -> None:
    """Gráfico temporal: proba_recesion vs. tiempo con umbral y target real."""
    os.makedirs(os.path.dirname(RUTA_SALIDA_FIG), exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5))

    fechas = tabla["fecha"]
    proba = tabla["proba_recesion"]
    target = tabla["target_12m_real"]

    ax.plot(fechas, proba, "o-", color="#1f77b4", label="P(recesión)", linewidth=1.8)
    ax.axhline(umbral, color="#d62728", linestyle="--",
               label=f"Umbral OOF ({umbral:.3f})")

    mascara_pos = target == 1
    if mascara_pos.any():
        ax.scatter(
            fechas[mascara_pos], proba[mascara_pos],
            color="#2ca02c", s=120, zorder=5,
            label="target_12m = 1 (anticipa COVID)",
        )

    inicio_recesion_anticipada = pd.Timestamp("2019-02-28")
    fin_recesion_anticipada = pd.Timestamp("2019-04-30")
    ax.axvspan(
        inicio_recesion_anticipada, fin_recesion_anticipada,
        color="#ffbf66", alpha=0.25,
        label="Ventana anticipatoria NBER COVID",
    )

    ax.set_title(
        "Anticipación del modelo a la recesión COVID (Tarea 2.10)\n"
        "Probabilidad predicha vs. umbral durante 2018-01 → 2020-02",
        fontsize=11,
    )
    ax.set_xlabel("Fecha")
    ax.set_ylabel("P(recesión a 12 meses)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(RUTA_SALIDA_FIG, dpi=150)
    plt.close(fig)


def main():
    print("=" * 70)
    print("ANÁLISIS DE ROBUSTEZ COVID — MODELO GANADOR")
    print("=" * 70)

    pipeline = joblib.load(RUTA_MODELO)
    umbral = cargar_umbral_ganador()
    X, _, y_12m = cargar_dataset()

    tabla = construir_tabla(X, y_12m, pipeline, umbral)
    resumen = resumen_estadistico(tabla, umbral)

    print(f"Ventana analizada: {resumen['ventana_inicio']} → {resumen['ventana_fin']}")
    print(f"Observaciones: {resumen['n_observaciones']}  |  Umbral: {umbral:.4f}")
    print(f"\nPositivos target_12m (anticipan COVID): "
          f"{resumen['positivos_target_12m']['n']}")
    if resumen['positivos_target_12m']['n'] > 0:
        pos = resumen['positivos_target_12m']
        print(f"  Fechas: {pos['fechas']}")
        print(f"  P(recesión): media={pos['proba_media']:.4f}  "
              f"max={pos['proba_max']:.4f}  min={pos['proba_min']:.4f}")
        print(f"  Detectados: {pos['detectados']} / {pos['n']}  "
              f"(no detectados: {pos['no_detectados']})")

    neg = resumen['negativos_target_12m']
    print(f"\nNegativos target_12m: {neg['n']}")
    print(f"  P(recesión): media={neg['proba_media']:.4f}  max={neg['proba_max']:.4f}")
    print(f"  Falsos positivos: {neg['falsos_positivos']}")

    print(f"\nProba máxima en la ventana: {resumen['max_proba_global_ventana']:.4f} "
          f"({resumen['fecha_max_proba']})")

    generar_grafico(tabla, umbral)
    print(f"\nFigura guardada en: {RUTA_SALIDA_FIG}")

    os.makedirs(os.path.dirname(RUTA_SALIDA_JSON), exist_ok=True)
    salida = {
        "resumen": resumen,
        "tabla_mensual": tabla.assign(
            fecha=tabla["fecha"].dt.strftime("%Y-%m-%d")
        ).to_dict(orient="records"),
    }
    with open(RUTA_SALIDA_JSON, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)
    print(f"Resultados persistidos en: {RUTA_SALIDA_JSON}")

    print("\n" + "-" * 70)
    print("Tabla mensual completa:")
    print(tabla.to_string(index=False))


if __name__ == "__main__":
    main()
