import pytest
import pandas as pd
from modulo_1_data.spread_builder import SpreadBuilder

# ============================================================================
# FIXTURES - Preparación de datos reutilizables
# ============================================================================

@pytest.fixture
def spread_builder():
    return SpreadBuilder()

@pytest.fixture
def datos_credito():
    # gs10: bonos tesoro, baa: bonos corporativos
    return pd.DataFrame({
        'baa': [7.0, 8.5, 6.0],
        'gs10': [4.0, 5.0, 4.0]
    })

# ============================================================================
# TESTS - Siguiendo el patrón AAA
# ============================================================================

def test_calcular_credit_spread_exitoso(spread_builder, datos_credito):
    # ARRANGE
    # Cálculo esperado: baa - gs10
    esperado = pd.Series([3.0, 3.5, 2.0], name='credit_spread')
    
    # ACT
    # Ejecuta el cálculo de diferenciales sobre el DataFrame
    resultado = spread_builder.calcular_spreads(datos_credito)
    
    # ASSERT
    # Verifica la existencia de la columna calculada
    assert 'credit_spread' in resultado.columns
    
    # Compara valores resultantes con la serie esperada
    pd.testing.assert_series_equal(
        resultado['credit_spread'], 
        esperado, 
        check_names=False
    )