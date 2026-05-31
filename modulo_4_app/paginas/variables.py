"""Página Variables — diccionario de los 16 indicadores del modelo.

Las variables se agrupan en cinco categorías económicas. Cada ficha contiene
metadata técnica (fuente, publicador, unidad, frecuencia y transformaciones),
una breve justificación económica de su relevancia para anticipar recesiones
y un gráfico histórico Plotly con los periodos de recesión NBER superpuestos.
La información proviene del Anexo D y §2.2.2 de `memoria.md`.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from componentes.carga_datos import cargar_dataset
from componentes.estilos import COLOR_NBER, OPACIDAD_BANDAS_NBER

COLOR_LINEA = "#1E3A8A"


CATEGORIAS = [
    {
        "titulo": "🧑‍💼 Mercado laboral",
        "intro": (
            "Indicadores del nivel de empleo y del flujo hacia el desempleo. "
            "Capturan el deterioro del mercado de trabajo, que tiende a "
            "acelerarse abruptamente en las transiciones hacia recesión."
        ),
        "variables": [
            {
                "id": "unrate",
                "nombre": "Tasa de desempleo civil",
                "fuente": "FRED · serie `UNRATE`",
                "publicador": "U.S. Bureau of Labor Statistics (BLS)",
                "unidad": "Porcentaje (%)",
                "frecuencia": "Mensual",
                "transformaciones": "Ninguna (uso en niveles); resampleo a fin de mes",
                "porque_importa": (
                    "El desempleo es un indicador rezagado pero crítico del "
                    "ciclo: aumenta abruptamente al inicio de las recesiones "
                    "y su persistencia condiciona la profundidad de la "
                    "contracción. La **regla de Sahm** (subida de 0,5 pp en "
                    "la media móvil de 3 meses sobre su mínimo del último "
                    "año) ha identificado correctamente todas las recesiones "
                    "NBER desde 1970."
                ),
            },
            {
                "id": "icsa",
                "nombre": "Solicitudes iniciales de subsidio de desempleo",
                "fuente": "FRED · serie `ICSA`",
                "publicador": "U.S. Department of Labor",
                "unidad": "Número de solicitudes",
                "frecuencia": "Semanal (nativa) → mensual",
                "transformaciones": "Resampleo semanal → mensual mediante último viernes",
                "porque_importa": (
                    "Indicador de **alta frecuencia** que captura el "
                    "deterioro laboral antes que la tasa de desempleo. Las "
                    "subidas pronunciadas en el flujo de solicitudes iniciales "
                    "suelen anticipar contracciones de empleo, ya que reflejan "
                    "despidos en tiempo real y no requieren tiempo de búsqueda "
                    "para registrarse como desempleo formal."
                ),
            },
        ],
    },
    {
        "titulo": "💰 Precios e inflación",
        "intro": (
            "Indicadores de inflación al consumidor, al productor y de "
            "materias primas estratégicas. La política monetaria responde a "
            "estos niveles y los shocks de oferta sobre el petróleo han "
            "precedido la mayoría de recesiones modernas."
        ),
        "variables": [
            {
                "id": "cpiaucsl",
                "nombre": "Índice de precios al consumidor (IPC)",
                "fuente": "FRED · serie `CPIAUCSL`",
                "publicador": "U.S. Bureau of Labor Statistics (BLS)",
                "unidad": "Índice (1982-1984 = 100)",
                "frecuencia": "Mensual",
                "transformaciones": "Ninguna (uso en niveles del índice)",
                "porque_importa": (
                    "La inflación está históricamente ligada al ciclo "
                    "monetario: episodios de inflación alta provocan subidas "
                    "agresivas de tipos por parte de la Reserva Federal "
                    "(Volcker 1980, 2022-2023), que a su vez precipitan "
                    "recesiones por el enfriamiento de la demanda agregada."
                ),
            },
            {
                "id": "ppiaco",
                "nombre": "Índice de precios al productor (IPP)",
                "fuente": "FRED · serie `PPIACO`",
                "publicador": "U.S. Bureau of Labor Statistics (BLS)",
                "unidad": "Índice (1982 = 100)",
                "frecuencia": "Mensual",
                "transformaciones": "Ninguna (uso en niveles)",
                "porque_importa": (
                    "Indicador **adelantado del IPC**: las presiones de "
                    "precios en la cadena productiva (materias primas, "
                    "bienes intermedios) se trasladan al consumidor final "
                    "con un retardo típico de varios meses. Captura "
                    "tensiones inflacionarias antes de que sean visibles "
                    "en la cesta del consumidor."
                ),
            },
            {
                "id": "wtisplc",
                "nombre": "Precio del crudo West Texas Intermediate (WTI)",
                "fuente": "FRED · serie `WTISPLC`",
                "publicador": "U.S. Energy Information Administration (EIA)",
                "unidad": "Dólares (USD) por barril",
                "frecuencia": "Mensual",
                "transformaciones": "Ninguna (uso en niveles)",
                "porque_importa": (
                    "Los **shocks petroleros** han precedido prácticamente "
                    "todas las recesiones estadounidenses desde 1973 "
                    "(Hamilton, 1983, 2003). Subidas bruscas del precio del "
                    "crudo encarecen el transporte y la producción, "
                    "comprimiendo márgenes empresariales y renta disponible "
                    "de los hogares."
                ),
            },
            {
                "id": "precio_oro",
                "nombre": "Precio del oro (XAU/USD)",
                "fuente": "Empalme LBMA (1968-1999) + Yahoo Finance / CME (2000-presente)",
                "publicador": "LBMA y CME Group",
                "unidad": "Dólares (USD) por onza troy",
                "frecuencia": "Mensual",
                "transformaciones": "Empalme temporal entre series; resampleo a fin de mes",
                "porque_importa": (
                    "Activo **refugio por excelencia**. Subidas pronunciadas "
                    "del precio del oro reflejan aversión al riesgo, "
                    "incertidumbre macroeconómica o pérdida de confianza en "
                    "los activos financieros tradicionales, todas señales "
                    "típicas de la fase previa a un cambio de ciclo."
                ),
            },
        ],
    },
    {
        "titulo": "🏭 Actividad real e indicadores líderes",
        "intro": (
            "Indicadores de la economía real (producción, vivienda) y un "
            "indicador líder compuesto diseñado explícitamente para "
            "anticipar puntos de giro del ciclo."
        ),
        "variables": [
            {
                "id": "indpro",
                "nombre": "Índice de producción industrial",
                "fuente": "FRED · serie `INDPRO`",
                "publicador": "Board of Governors of the Federal Reserve System",
                "unidad": "Índice (2017 = 100)",
                "frecuencia": "Mensual",
                "transformaciones": "Ninguna (uso en niveles del índice)",
                "porque_importa": (
                    "Componente clave de los **indicadores coincidentes** del "
                    "ciclo económico. Caídas sostenidas señalan contracción "
                    "real de la actividad manufacturera, minera y de servicios "
                    "públicos, sectores intensivos en capital muy sensibles "
                    "a las condiciones financieras."
                ),
            },
            {
                "id": "houst",
                "nombre": "Inicios de construcción de viviendas",
                "fuente": "FRED · serie `HOUST`",
                "publicador": "U.S. Census Bureau",
                "unidad": "Miles de unidades (anualizadas)",
                "frecuencia": "Mensual",
                "transformaciones": "Ninguna (uso en niveles)",
                "porque_importa": (
                    "La vivienda es uno de los componentes **más cíclicos** "
                    "del PIB y un indicador adelantado clásico. Como afirma "
                    "Leamer (2015), *housing IS the business cycle*: los "
                    "inicios de construcción reaccionan rápidamente a los "
                    "tipos hipotecarios y anticipan el resto de la "
                    "inversión privada."
                ),
            },
            {
                "id": "usalolitoaastsam",
                "nombre": "Indicador líder compuesto OECD para EE.UU.",
                "fuente": "FRED · serie `USALOLITOAASTSAM`",
                "publicador": "Organisation for Economic Co-operation and Development (OECD)",
                "unidad": "Índice (long-term avg = 100)",
                "frecuencia": "Mensual",
                "transformaciones": "Ninguna (uso en niveles del índice)",
                "porque_importa": (
                    "Indicador **diseñado específicamente para anticipar "
                    "puntos de giro del ciclo económico** con 6-9 meses de "
                    "antelación. Agrega información de múltiples fuentes "
                    "(manufactura, permisos de construcción, mercado "
                    "bursátil, encuestas) en un único índice estandarizado "
                    "y desestacionalizado por la OECD."
                ),
            },
        ],
    },
    {
        "titulo": "🏦 Política monetaria y mercado financiero",
        "intro": (
            "Variables que reflejan la postura monetaria de la Reserva "
            "Federal, los rendimientos del Tesoro y el comportamiento de los "
            "mercados bursátiles, todos forward-looking por construcción."
        ),
        "variables": [
            {
                "id": "m2sl",
                "nombre": "Agregado monetario M2",
                "fuente": "FRED · serie `M2SL`",
                "publicador": "Board of Governors of the Federal Reserve System",
                "unidad": "Miles de millones de dólares (USD)",
                "frecuencia": "Mensual",
                "transformaciones": "Ninguna (uso en niveles)",
                "porque_importa": (
                    "Mide la **liquidez en circulación** en la economía. "
                    "Contracciones reales de M2 (deflactadas por IPC) son "
                    "históricamente raras y se asocian a episodios de "
                    "tensión financiera severa. Friedman & Schwartz (1963) "
                    "establecieron el papel central de los agregados "
                    "monetarios en la dinámica del ciclo."
                ),
            },
            {
                "id": "gs10",
                "nombre": "Rendimiento del Tesoro a 10 años",
                "fuente": "FRED · serie `GS10`",
                "publicador": "Board of Governors of the Federal Reserve System",
                "unidad": "Porcentaje (%) anualizado",
                "frecuencia": "Mensual",
                "transformaciones": "Ninguna; también input de `yield_spread` y `credit_spread`",
                "porque_importa": (
                    "Refleja las **expectativas del mercado sobre crecimiento "
                    "e inflación a largo plazo**. Caídas pronunciadas "
                    "anticipan ralentización económica esperada y/o "
                    "expectativas de relajación monetaria futura por parte "
                    "de la Reserva Federal."
                ),
            },
            {
                "id": "tb3ms",
                "nombre": "Tipo de la letra del Tesoro a 3 meses",
                "fuente": "FRED · serie `TB3MS`",
                "publicador": "Board of Governors of the Federal Reserve System",
                "unidad": "Porcentaje (%) anualizado",
                "frecuencia": "Mensual",
                "transformaciones": "Ninguna; también input de `yield_spread`",
                "porque_importa": (
                    "Refleja directamente la **política monetaria de corto "
                    "plazo** de la Reserva Federal. Cuando el tipo a 3 meses "
                    "supera al tipo a 10 años (inversión de la curva de "
                    "tipos) el mercado anticipa una contracción económica "
                    "que justificará bajadas futuras de tipos."
                ),
            },
            {
                "id": "sp500",
                "nombre": "Índice bursátil S&P 500",
                "fuente": "Yahoo Finance · ticker `^GSPC`",
                "publicador": "S&P Dow Jones Indices",
                "unidad": "Puntos del índice",
                "frecuencia": "Diaria (nativa) → mensual",
                "transformaciones": "Cierre del último día de cotización del mes",
                "porque_importa": (
                    "Los mercados bursátiles son **forward-looking por "
                    "construcción**: los precios incorporan expectativas "
                    "sobre beneficios empresariales futuros. Caídas "
                    "sostenidas anticipan menores beneficios esperados, "
                    "habitualmente asociados a desaceleraciones económicas."
                ),
            },
        ],
    },
    {
        "titulo": "📉 Crédito y spreads construidos",
        "intro": (
            "Variables que capturan el coste del crédito corporativo y dos "
            "diferenciales calculados (yield spread y credit spread) que "
            "son los predictores de recesiones más documentados en la "
            "literatura macroeconómica."
        ),
        "variables": [
            {
                "id": "baa",
                "nombre": "Rendimiento de bonos corporativos Moody's Baa",
                "fuente": "FRED · serie `BAA`",
                "publicador": "Moody's Investors Service",
                "unidad": "Porcentaje (%) anualizado",
                "frecuencia": "Mensual",
                "transformaciones": "Ninguna; también input de `credit_spread`",
                "porque_importa": (
                    "Rendimiento de bonos corporativos de **grado de "
                    "inversión medio** (Baa, equivalente a BBB de S&P). Su "
                    "tensión refleja el deterioro percibido de las "
                    "condiciones crediticias y del riesgo empresarial. "
                    "Sirve como input directo del `credit_spread`."
                ),
            },
            {
                "id": "yield_spread",
                "nombre": "Pendiente de la curva de tipos (calculado: GS10 − TB3MS)",
                "fuente": "Cálculo propio sobre series FRED `GS10` y `TB3MS`",
                "publicador": "Construido por el proyecto",
                "unidad": "Puntos porcentuales (pp)",
                "frecuencia": "Mensual",
                "transformaciones": "Resta directa de las dos series tras alineación a fin de mes",
                "porque_importa": (
                    "El **predictor de recesiones más documentado** de la "
                    "literatura macroeconómica. Una inversión de la curva "
                    "(valor negativo) ha precedido **todas las recesiones "
                    "NBER desde 1969** sin excepción (Estrella & Mishkin, "
                    "1998), con una ventana típica de 6 a 18 meses "
                    "perfectamente coherente con el `target_12m` del modelo."
                ),
            },
            {
                "id": "credit_spread",
                "nombre": "Prima de riesgo de crédito corporativo (calculado: BAA − GS10)",
                "fuente": "Cálculo propio sobre series FRED `BAA` y `GS10`",
                "publicador": "Construido por el proyecto",
                "unidad": "Puntos porcentuales (pp)",
                "frecuencia": "Mensual",
                "transformaciones": "Resta directa de las dos series tras alineación a fin de mes",
                "porque_importa": (
                    "Diferencial entre el rendimiento corporativo Baa y el "
                    "Treasury a 10 años. Capta **deterioro crediticio y "
                    "aversión al riesgo** del sector empresarial. Gilchrist "
                    "& Zakrajšek (2012) demostraron su elevado poder "
                    "predictivo sobre la actividad económica futura."
                ),
            },
        ],
    },
]


@st.cache_data(show_spinner=False)
def _calcular_periodos_recesion() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Devuelve los periodos contiguos de recesión NBER del dataset."""
    df = cargar_dataset()
    usrec = df["usrec"]
    periodos: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    en_recesion = False
    inicio: pd.Timestamp | None = None
    fecha_anterior: pd.Timestamp | None = None
    for fecha, valor in usrec.items():
        if valor == 1 and not en_recesion:
            inicio = fecha
            en_recesion = True
        elif valor == 0 and en_recesion:
            periodos.append((inicio, fecha_anterior))
            en_recesion = False
        fecha_anterior = fecha
    if en_recesion and inicio is not None and fecha_anterior is not None:
        periodos.append((inicio, fecha_anterior))
    return periodos


def _construir_grafico(columna: str) -> go.Figure:
    """Construye el gráfico Plotly de una variable con bandas NBER superpuestas."""
    df = cargar_dataset()
    serie = df[columna].dropna()
    periodos = _calcular_periodos_recesion()

    fig = go.Figure()
    for inicio, fin in periodos:
        fig.add_vrect(
            x0=inicio, x1=fin,
            fillcolor=COLOR_NBER, opacity=OPACIDAD_BANDAS_NBER,
            layer="below", line_width=0,
        )
    fig.add_trace(go.Scatter(
        x=serie.index, y=serie.values,
        mode="lines",
        line=dict(color=COLOR_LINEA, width=1.5),
        hovertemplate=f"<b>%{{x|%Y-%m}}</b><br>{columna}=%{{y:.3f}}<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=280,
        plot_bgcolor="white",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EAEAEA")
    fig.update_yaxes(showgrid=True, gridcolor="#EAEAEA")
    return fig


def _render_variable(var: dict) -> None:
    """Renderiza una ficha de variable dentro de un expander."""
    with st.expander(f"`{var['id']}` — {var['nombre']}"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Fuente:** {var['fuente']}")
            st.markdown(f"**Publicador:** {var['publicador']}")
            st.markdown(f"**Unidad:** {var['unidad']}")
        with col_b:
            st.markdown(f"**Frecuencia nativa:** {var['frecuencia']}")
            st.markdown(f"**Transformaciones:** {var['transformaciones']}")
        st.markdown("**¿Por qué importa para predecir recesiones?**")
        st.markdown(var["porque_importa"])
        st.plotly_chart(
            _construir_grafico(var["id"]),
            use_container_width=True,
            config={"displayModeBar": False},
            key=f"grafico_{var['id']}",
        )
        st.caption(
            "Serie histórica 1967-2025. Las bandas grises marcan los periodos "
            "de recesión NBER (`usrec = 1`)."
        )


def renderizar():
    """Renderiza la página Variables."""
    st.title("📚 Variables del modelo")
    st.caption(
        "Diccionario de los 16 indicadores macro-financieros que alimentan el modelo."
    )
    st.divider()

    st.markdown(
        "El modelo final utiliza **16 variables predictoras** seleccionadas "
        "por su respaldo en la literatura macroeconómica como anticipadores "
        "del ciclo económico estadounidense. Todas las series proceden de "
        "fuentes públicas (FRED y Yahoo Finance) y se persisten en "
        "frecuencia mensual sobre el periodo **1967-2025**. Las fichas "
        "técnicas completas se encuentran en el **Anexo D** de la memoria "
        "del proyecto."
    )

    st.info(
        "💡 Las variables se agrupan en **5 categorías económicas**. "
        "Despliega cada ficha para ver el detalle técnico y la "
        "justificación económica de cada indicador."
    )

    st.divider()

    for cat in CATEGORIAS:
        st.subheader(cat["titulo"])
        st.markdown(cat["intro"])
        for var in cat["variables"]:
            _render_variable(var)
        st.divider()

    st.caption(
        "📖 Referencias clave: Estrella & Mishkin (1998), Gilchrist & "
        "Zakrajšek (2012), Hamilton (1983, 2003), Leamer (2015), Friedman "
        "& Schwartz (1963). Bibliografía completa en la memoria del "
        "proyecto y en la página *Acerca de*."
    )
