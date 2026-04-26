import pytest
import pandas as pd
import os
from modulo_1_data.preprocessor import DataPreprocessor

# ============================================================================
# FIXTURES - Preparación de datos y archivos temporales
# ============================================================================

@pytest.fixture
def preprocesador(tmpdir):
    # Crea el preprocesador apuntando al directorio temporal de pruebas
    return DataPreprocessor(ruta_bruta=str(tmpdir))

@pytest.fixture
def mock_sp500_diario(tmpdir):
    # ARRANGE: Definir ruta y estructura de cabecera tipo Yahoo Finance
    ruta_archivo = os.path.join(str(tmpdir), "sp500.csv")
    
    # Escribir las 2 líneas de cabecera (skiprows=2 en el script original)
    with open(ruta_archivo, 'w') as f:
        f.write("Price,Close,High,Low,Open,Volume\n")
        f.write("Ticker,^GSPC,^GSPC,^GSPC,^GSPC,^GSPC\n")
    
    # Datos diarios con cierres específicos para enero y febrero 2020
    datos = {
        'Date': [
            '2020-01-02', '2020-01-15', '2020-01-31', 
            '2020-02-03', '2020-02-14', '2020-02-28'
        ],
        'Close': [3000.0, 3100.0, 3200.0, 3250.0, 3300.0, 3350.0]
    }
    df_diario = pd.DataFrame(datos).set_index('Date')
    
    # Guardar los datos añadiéndolos después de la cabecera
    df_diario.to_csv(ruta_archivo, mode='a')
    return ruta_archivo

# ============================================================================
# TESTS - Validación de carga y resampling mensual (ME)
# ============================================================================

def test_load_sp500_resampling_mensual(preprocesador, mock_sp500_diario):
    # ARRANGE
    # Definimos los valores esperados (el último cierre registrado de cada mes)
    valor_esperado_enero = 3200.0
    valor_esperado_febrero = 3350.0
    
    # ACT
    # El método carga, salta 2 filas, convierte a datetime y aplica .resample('ME').last()
    resultado = preprocesador.load_sp500()
    
    # ASSERT
    # 1. Verificar nombre de la serie resultante
    assert resultado.name == 'sp500'
    
    # 2. Verificar longitud (se reducen 6 días a 2 meses)
    assert len(resultado) == 2
    
    # 3. Verificar que los valores corresponden al ÚLTIMO dato de cada mes
    assert resultado.iloc[0] == valor_esperado_enero
    assert resultado.iloc[1] == valor_esperado_febrero
    
    # 4. Verificar que el índice resultante es fin de mes (Month End)
    # Comprobamos que el día coincide con el último día posible de ese mes
    assert all(resultado.index.day == resultado.index.days_in_month)