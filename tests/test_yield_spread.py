import pytest
import pandas as pd
from modulo_1_data.spread_builder import SpreadBuilder

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def spread_builder():
    return SpreadBuilder()


@pytest.fixture
def datos_tipos():
    # Datos para yield_spread: gs10 - tb3ms
    return pd.DataFrame({
        'gs10': [5.0, 6.0, 4.5],
        'tb3ms': [3.0, 4.0, 3.5]
    })

# ============================================================================
# TESTS
# ============================================================================

def test_calcular_yield_spread(spread_builder, datos_tipos):
    # ARRANGE - Resultado esperado: gs10 - tb3ms
    esperado = pd.Series([2.0, 2.0, 1.0], name='yield_spread')
    
    # ACT
    resultado = spread_builder.calcular_spreads(datos_tipos)
    
    # ASSERT - Verifica columna y valores
    assert 'yield_spread' in resultado.columns
    pd.testing.assert_series_equal(
        resultado['yield_spread'], 
        esperado, 
        check_names=False
    )