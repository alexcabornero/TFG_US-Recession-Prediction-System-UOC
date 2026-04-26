import pytest
import pandas as pd
import numpy as np
import os
from modulo_1_data.target_builder import TargetBuilder

# ============================================================================
# FIXTURES - Preparación de datos y archivos temporales
# ============================================================================

@pytest.fixture
def target_builder(tmpdir):
    # Instancia de TargetBuilder apuntando al directorio temporal
    return TargetBuilder(ruta_procesada=str(tmpdir))

@pytest.fixture
def datos_maestros_simulados(tmpdir):
    # ARRANGE: Crear 15 meses de datos para permitir shift de 12 sin agotar el DF
    fechas = pd.date_range("2020-01-01", periods=15, freq='ME')
    
    # usrec: 5 meses 0, 3 meses 1 (recesión), 7 meses 0
    usrec_values = [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    
    df = pd.DataFrame({
        'usrec': usrec_values,
        'feature1': np.random.rand(15),
        'feature2': np.random.rand(15)
    }, index=fechas)
    
    # Guardar como dataset_maestro.csv en la ruta temporal
    ruta_archivo = os.path.join(str(tmpdir), "dataset_maestro.csv")
    df.to_csv(ruta_archivo)
    return df

# ============================================================================
# TESTS - Validación de targets (Patrón AAA)
# ============================================================================

def test_construir_targets_desplazamiento_correcto(target_builder, datos_maestros_simulados):
    # ARRANGE
    # El valor de usrec en el mes 7 (julio 2020) es 1
    # El valor de usrec en el mes 13 (enero 2021) es 0
    
    # ACT
    resultado = target_builder.construir_targets()

    # ASSERT
    # Verificar que las columnas de target existen
    assert 'target_6m' in resultado.columns
    assert 'target_12m' in resultado.columns
    
    # Verificar que enero 2020 tiene el target de julio 2020 (t+6)
    assert resultado.loc['2020-01-31', 'target_6m'] == 1
    
    # Verificar que enero 2020 tiene el target de enero 2021 (t+12)
    assert resultado.loc['2020-01-31', 'target_12m'] == 0
    
    # Verificar longitud: 15 meses - 12 (máximo desplazamiento) = 3 filas
    assert len(resultado) == 3

def test_construir_targets_sin_nulos(target_builder, datos_maestros_simulados):
    # ARRANGE
    # Datos preparados por la fixture

    # ACT
    resultado = target_builder.construir_targets()

    # ASSERT
    # Verificar que no queda ningún valor nulo tras el dropna interno
    assert resultado.isnull().sum().sum() == 0
    assert len(resultado) < len(datos_maestros_simulados)

def test_construir_targets_alineacion_temporal(target_builder, datos_maestros_simulados):
    # ARRANGE
    # Datos preparados por la fixture

    # ACT
    resultado = target_builder.construir_targets()
    features = resultado.drop(columns=['target_6m', 'target_12m'])

    # ASSERT
    # Verificar que los índices de features y targets coinciden exactamente
    pd.testing.assert_index_equal(features.index, resultado['target_6m'].index)
    pd.testing.assert_index_equal(features.index, resultado['target_12m'].index)

def test_construir_targets_valores_binarios(target_builder, datos_maestros_simulados):
    # ARRANGE
    # Datos preparados por la fixture

    # ACT
    resultado = target_builder.construir_targets()

    # ASSERT
    # Verificar que los targets solo contienen 0 o 1 (clases discretas)
    assert set(resultado['target_6m'].unique()).issubset({0, 1})
    assert set(resultado['target_12m'].unique()).issubset({0, 1})