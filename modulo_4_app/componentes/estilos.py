"""Constantes visuales y de paleta semántica de la aplicación.

Reutiliza los colores empleados en las figuras matplotlib del Hito 2 para
mantener coherencia visual entre los assets pre-generados y los gráficos
Plotly que renderice la app en runtime.
"""

COLOR_PROBABILIDAD = "#1f77b4"
COLOR_UMBRAL = "#d62728"
COLOR_NBER = "#888888"
COLOR_VERDE = "#2ca02c"
COLOR_NARANJA = "#ffbf66"

OPACIDAD_BANDAS_NBER = 0.20

UMBRAL_DECISION = 0.6543427530031798
UMBRAL_TENSION = 0.30


def estado_probabilidad(proba: float) -> tuple[str, str, str]:
    """Devuelve etiqueta, color y emoji para una probabilidad dada.

    Args:
        proba: Probabilidad de recesión predicha por el modelo, en [0, 1].

    Returns:
        Tupla (etiqueta, color_hex, emoji) según los rangos definidos en 3.1.3.
    """
    if proba < UMBRAL_TENSION:
        return ("Sin alerta", COLOR_VERDE, "✅")
    if proba < UMBRAL_DECISION:
        return ("Tensión", COLOR_NARANJA, "⚠️")
    return ("Alerta de recesión", COLOR_UMBRAL, "🚨")
