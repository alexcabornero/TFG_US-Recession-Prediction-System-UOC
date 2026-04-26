import pytest
import pandas as pd
import os
from modulo_1_data.preprocessor import DataPreprocessor

# ============================================================================
# FIXTURES - Preparación de archivos y datos simulados
# ============================================================================

@pytest.fixture
def preprocesador(tmpdir):
    # Instancia el preprocesador en un directorio temporal para pruebas
    return DataPreprocessor(ruta_bruta=str(tmpdir))

@pytest.fixture
def mock_dataset_completo(tmpdir):
    # ARRANGE: Definir fechas de fin de mes para 2020
    fechas = pd.date_range("2020-01-01", periods=3, freq='ME')
    
    # 1. Crear 3 series FRED básicas
    series_fred = {
        "usrec": [0.0, 0.0, 1.0],
        "unrate": [3.5, 3.5, 4.4],
        "cpiaucsl": [258.0, 258.6, 258.1]
    }
    for nombre, valores in series_fred.items():
        ruta = os.path.join(str(tmpdir), f"{nombre}.csv")
        pd.DataFrame({"valor": valores}, index=fechas).to_csv(ruta)

    # 2. Crear sp500.csv (formato Yahoo con 2 líneas de cabecera)
    ruta_sp500 = os.path.join(str(tmpdir), "sp500.csv")
    with open(ruta_sp500, 'w') as f:
        f.write("Price,Close,High,Low,Open,Volume\n")
        f.write("Ticker,^GSPC,^GSPC,^GSPC,^GSPC,^GSPC\n")
    df_sp500 = pd.DataFrame({"Close": [3225.0, 2954.0, 2584.0]}, index=fechas)
    df_sp500.to_csv(ruta_sp500, mode='a')

    # 3. Crear Oro (histórico y moderno)
    # Incluimos un NaN en el moderno para validar que la alineación lo maneja/elimina
    df_oro_hist = pd.DataFrame({"Price": [1517.0, 1583.0, 1591.0]}, index=fechas)
    df_oro_hist.to_csv(os.path.join(str(tmpdir), "oro_historico.csv"))
    
    ruta_oro_mod = os.path.join(str(tmpdir), "oro.csv")
    with open(ruta_oro_mod, 'w') as f:
        f.write("Price,Close,...\n")
        f.write("Ticker,GC=F,...\n")
    # Fila de febrero como NaN para forzar la limpieza de alinear_todas_las_series
    df_oro_mod = pd.DataFrame({"Close": [1517.0, None, 1591.0]}, index=fechas)
    df_oro_mod.to_csv(ruta_oro_mod, mode='a')

    return tmpdir

# ============================================================================
# TESTS - Validación de alineación, limpieza y frecuencia
# ============================================================================

def test_alinear_todas_las_series_basico(preprocesador, mock_dataset_completo):
    # ACT
    # Ejecuta la carga y alineación de todas las fuentes
    resultado = preprocesador.alinear_todas_las_series()

    # ASSERT
    # 1. Verificar que es un DataFrame
    assert isinstance(resultado, pd.DataFrame)
    
    # 2. Verificar columnas esperadas (3 FRED + 2 Mercado)
    columnas = resultado.columns
    assert all(col in columnas for col in ['usrec', 'unrate', 'cpiaucsl', 'sp500', 'precio_oro'])
    
    # 3. Verificar que no hay NaNs (el NaN de oro en febrero debe haber causado dropna)
    assert resultado.isnull().sum().sum() == 0
    
    # 4. Verificar frecuencia mensual estricta (Month End)
    assert resultado.index.freqstr in ['ME', 'M']
    assert all(resultado.index.day == resultado.index.days_in_month)

def test_alinear_todas_las_series_con_fecha_inicio(preprocesador, mock_dataset_completo):
    # ARRANGE
    fecha_filtro = '2020-02-01'
    
    # ACT
    # Se solicita el dataset a partir de febrero
    resultado = preprocesador.alinear_todas_las_series(fecha_inicio=fecha_filtro)

    # ASSERT
    # 1. Verificar que el primer registro es igual o posterior a la fecha filtro
    assert resultado.index.min() >= pd.to_datetime(fecha_filtro)
    
    # 2. Verificar que se mantienen las propiedades de limpieza
    assert resultado.isnull().sum().sum() == 0