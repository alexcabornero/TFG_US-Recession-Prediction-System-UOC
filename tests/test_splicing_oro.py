import pytest
import pandas as pd
import os
from modulo_1_data.preprocessor import DataPreprocessor

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def preprocesador(tmpdir):
    # Crea el preprocesador apuntando a un directorio temporal
    return DataPreprocessor(ruta_bruta=str(tmpdir))

@pytest.fixture
def mock_oro_files(tmpdir):
    # Datos históricos (1999-10 a 2000-03) - cruzan el corte 2000-01-01
    fechas_hist = pd.date_range("1999-10-01", periods=6, freq='ME')
    df_hist = pd.DataFrame({'Price': [280.0, 285.0, 290.0, 295.0, 300.0, 305.0]}, index=fechas_hist)
    ruta_hist = os.path.join(str(tmpdir), "oro_historico.csv")
    df_hist.to_csv(ruta_hist)

    # Datos modernos (2000-01 a 2000-06) con cabecera Yahoo
    fechas_mod = pd.date_range("2000-01-01", periods=6, freq='ME')
    ruta_mod = os.path.join(str(tmpdir), "oro.csv")
    with open(ruta_mod, 'w') as f:
        f.write("Price,Ticker,Date\n")
        f.write("Close,GC=F,None\n")
    
    df_mod = pd.DataFrame({'Close': [310.0, 315.0, 320.0, 325.0, 330.0, 335.0]}, index=fechas_mod)
    df_mod.to_csv(ruta_mod, mode='a')
    
    return ruta_hist, ruta_mod

# ============================================================================
# TESTS
# ============================================================================

def test_load_merged_gold_splicing_logic(preprocesador, mock_oro_files):
    # ARRANGE - Valores esperados según la lógica de corte en 2000-01-01
    precio_octubre_1999 = 280.0  # Debe venir de histórico (< 2000)
    precio_enero_2000 = 310.0    # Debe venir de moderno (>= 2000)
    precio_junio_2000 = 335.0    # Debe venir de moderno

    # ACT
    resultado = preprocesador.load_merged_gold()

    # ASSERT
    assert resultado.name == 'precio_oro'
    assert len(resultado) == 9  # 9 meses: oct-1999 a jun-2000
    
    # Verifica datos históricos antes del corte
    assert resultado.loc['1999-10-31'] == precio_octubre_1999
    
    # Verifica prioridad de datos modernos desde 2000 en adelante
    assert resultado.loc['2000-01-31'] == precio_enero_2000
    assert resultado.loc['2000-06-30'] == precio_junio_2000
    
    # Verifica ausencia de nulos
    assert resultado.isnull().sum() == 0


def test_splicing_oro_sin_huecos(preprocesador, mock_oro_files):
    # ARRANGE - Verificar continuidad temporal sin huecos
    
    # ACT
    resultado = preprocesador.load_merged_gold()
    
    # ASSERT - Verifica que diciembre 1999 y enero 2000 están presentes (punto de unión)
    assert pd.Timestamp("1999-12-31") in resultado.index
    assert pd.Timestamp("2000-01-31") in resultado.index
    
    # Verifica continuidad: 9 meses consecutivos sin huecos
    fechas_esperadas = pd.date_range("1999-10-01", periods=9, freq='ME')
    assert all(resultado.index == fechas_esperadas)


def test_splicing_oro_sin_duplicados(preprocesador, mock_oro_files):
    # ARRANGE - Verificar que no hay fechas duplicadas
    
    # ACT
    resultado = preprocesador.load_merged_gold()
    
    # ASSERT - Verifica que el índice no tiene duplicados
    assert resultado.index.is_unique


def test_splicing_oro_sin_valores_nulos(preprocesador, mock_oro_files):
    # ARRANGE - Verificar ausencia total de NaNs
    
    # ACT
    resultado = preprocesador.load_merged_gold()
    
    # ASSERT - Ningún valor debe ser NaN
    assert resultado.notna().all()
    assert resultado.isnull().sum() == 0


def test_splicing_oro_cobertura_temporal_completa(preprocesador, mock_oro_files):
    # ARRANGE - Verificar que cubre todo el rango temporal esperado
    fecha_inicio_esperada = pd.Timestamp("1999-10-31")
    fecha_fin_esperada = pd.Timestamp("2000-06-30")
    
    # ACT
    resultado = preprocesador.load_merged_gold()
    
    # ASSERT - Verifica rangos de fechas
    assert resultado.index.min() == fecha_inicio_esperada
    assert resultado.index.max() == fecha_fin_esperada