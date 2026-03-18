@echo off
SETLOCAL ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION
TITLE Pipeline de Ingenieria de Datos - TFG


echo =======================================================
echo Iniciando Pipeline de Ingenieria de Datos (Hito 1)
echo =======================================================

:: Configurar el PYTHONPATH al directorio actual para resolver modulos internos
set "PYTHONPATH=%CD%"

echo [0/4] Verificando conectividad y credenciales (FRED/YF)...
python -m modulo_1_data.api_connector
if ERRORLEVEL 1 (echo Error de conexion o credenciales invalidas. Abortando. & pause & exit /b %ERRORLEVEL%)

echo [1/4] Descargando datos desde FRED y Yahoo Finance...
python -m modulo_1_data.downloader
if ERRORLEVEL 1 (echo Error en la descarga. Abortando. ;& pause ;& exit /b %ERRORLEVEL%)

echo [2/4] Preprocesando y alineando series temporales...
python -m modulo_1_data.preprocessor
if ERRORLEVEL 1 (echo Error en el preprocesamiento. Abortando. ;& pause ;& exit /b %ERRORLEVEL%)

echo [3/4] Construyendo spreads financieros...
python -m modulo_1_data.spread_builder
if ERRORLEVEL 1 (echo Error en el calculo de spreads. Abortando. ;& pause ;& exit /b %ERRORLEVEL%)

echo [4/4] Generando targets y dataset final...
python -m modulo_1_data.target_builder
if ERRORLEVEL 1 (echo Error en la generacion de targets. Abortando. ;& pause ;& exit /b %ERRORLEVEL%)

echo ======================================================
echo Pipeline completado con exito.
echo Dataset generado en: data/processed/dataset_final.csv
echo ======================================================
pause
