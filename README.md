# Sistema de Predicción de Crisis Financieras en EE.UU.

Este repositorio contiene el desarrollo de un sistema end-to-end de machine learning para la predicción de recesiones económicas en EE.UU., como parte de un Trabajo de Fin de Grado (TFG).

## Requisitos Previos

- Python >= 3.10
- Conexión a internet (para descarga de datos de FRED y Yahoo Finance)
- Clave de API de FRED (configurada en `.env`)

---

## 1. Configuración del Entorno (Manual)

Si es la primera vez que descarga el repositorio, debe configurar su entorno virtual de Python siguiendo estos pasos:

1. **Crear el entorno virtual:**
   ```powershell
   python -m venv venv
   ```

2. **Activar el entorno virtual:**
   ```powershell
   .\venv\Scripts\activate
   ```

3. **Instalar dependencias:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Configurar credenciales:**
   - Introduzca su `FRED_API_KEY` en el archivo `.env`.

---

## 2. Ejecución del Pipeline de Datos

Una vez configurado el entorno, puede ejecutar todo el flujo de Ingeniería de Datos (Hito 1) de forma automatizada:

1. Ejecutar el archivo **`ejecutar_ingenieria_datos.bat`** en la raíz del proyecto.
2. El script automáticamente:
   - Verificará la conectividad con las APIs.
   - Descargará los datos históricos.
   - Generará el dataset final en `data/processed/dataset_final.csv`.

---

## 3. Aplicación Web (Streamlit)

El sistema cuenta con una aplicación web interactiva que permite explorar el modelo, las variables predictoras, sus predicciones, su explicabilidad SHAP y los resultados de validación histórica y comparativa de modelos.

### Ejecutar en local

```powershell
streamlit run modulo_4_app/app.py
```

La aplicación arranca en `http://localhost:8501` con 6 páginas:

- **Overview** — estado actual del riesgo y descripción del proyecto.
- **Variables** — diccionario de los 16 indicadores agrupados por categoría económica con fichas técnicas, justificación económica y serie histórica con bandas NBER.
- **Predicción** — serie temporal de probabilidad de recesión con bandas NBER y gauge.
- **SHAP** — importancia global de indicadores y casos representativos VP / FN / VN.
- **Backtesting** — Walk-Forward CV (4 folds), hold-out segmentado (expansión / COVID) y comparativa de las 12 combinaciones modelo × rebalanceo.
- **Acerca de** — autor, disclaimer, referencias bibliográficas y licencia.

### Aplicación desplegada online

🔗 **URL pública:** https://tfg-crisis-financieras.streamlit.app/

### Desplegar en Streamlit Community Cloud

1. Hacer fork o clonar el repositorio en GitHub.
2. Entrar en [share.streamlit.io](https://share.streamlit.io) e iniciar sesión con la cuenta de GitHub.
3. Pulsar **New app** → seleccionar el repositorio, la rama (`master`) y el fichero `modulo_4_app/app.py` como entry point.
4. La app se despliega automáticamente; el tema, las dependencias y el modelo se leen del repositorio.
5. Actualizar el enlace público en este README.

---

## Estructura del Proyecto

```
.streamlit/
└── config.toml                    # Tema de la app (azul)
data/
├── raw/                           # CSVs descargados (gitignored)
└── processed/                     # Dataset maestro y derivados
docs/
└── figures/                       # Gráficos SHAP y análisis COVID
models/
├── final_model.pkl                # Modelo final serializado (versionado)
└── *.json                         # Métricas, splits y resultados
modulo_1_data/                     # Adquisición y preprocesamiento
modulo_2_modelado/                 # Modelado, validación y explicabilidad SHAP
modulo_3_explicabilidad/           # Otras alternativas futuras de explicabilidad
modulo_4_app/                      # Aplicación Streamlit
├── app.py                         # Entry point
├── paginas/                       # 6 páginas
└── componentes/                   # Estilos, sidebar y cargadores cacheados
requirements.txt                   # Dependencias con versiones fijadas
```

---

> [!NOTE]
> Este proyecto tiene fines exclusivamente académicos y no constituye asesoramiento financiero ni recomendaciones de inversión.
