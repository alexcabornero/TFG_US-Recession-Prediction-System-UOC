# Sistema de Predicción de Crisis Financieras en EE.UU.

Este repositorio contiene el desarrollo de un sistema end-to-end de machine learning para la predicción de recesiones económicas en EE.UU., como parte del Trabajo de Fin de Grado (TFG).

## Requisitos Previos

- Python >= 3.10
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
En caso de tener algún problema con pip install usar:
```powershell
   py -m pip install -r requirements.txt
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

## Estructura del Proyecto

- `module_1_data/`: Adquisición y preprocesamiento de datos.
- `data/`: Almacenamiento de datos brutos y procesados.

> [!NOTE]
> Este proyecto tiene fines exclusivamente académicos y no constituye asesoramiento financiero.
