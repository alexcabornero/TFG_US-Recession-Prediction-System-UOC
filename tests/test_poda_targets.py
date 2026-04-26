import pytest
import pandas as pd
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
def datos_poda_15_meses(tmpdir):
    # ARRANGE: Crear 15 meses exactos (Ene-2020 a Mar-2021)
    fechas = pd.date_range("2020-01-01", periods=15, freq='ME')
    
    # usrec con todos los valores en 0 para simplificar el enfoque en la poda
    df = pd.DataFrame({
        'usrec': [0] * 15,
        'feature_test': [1.0] * 15
    }, index=fechas)
    
    # Guardar archivo maestro simulado
    ruta_archivo = os.path.join(str(tmpdir), "dataset_maestro.csv")
    df.to_csv(ruta_archivo)
    return df

# ============================================================================
# TESTS - Validación de la Poda de Registros Finales
# ============================================================================

def test_construir_targets_poda_registros_finales(target_builder, datos_poda_15_meses):
    # ARRANGE
    # La fecha del último registro original es 2021-03-31
    ultimo_indice_original = datos_poda_15_meses.index[-1]
    
    # ACT
    # El método aplica shift(-6) y shift(-12), luego dropna()
    resultado = target_builder.construir_targets()

    # ASSERT
    # 1. Con shift(-12), solo quedan 3 registros válidos (15 iniciales - 12 perdidos)
    assert len(resultado) == 3
    
    # 2. Verificar que no existen valores nulos en las columnas de target
    assert resultado['target_6m'].isnull().sum() == 0
    assert resultado['target_12m'].isnull().sum() == 0
    
    # 3. Verificar que el último índice del resultado es 12 meses antes del original
    # El último registro debería ser 2020-03-31 (12 meses antes de 2021-03-31)
    fecha_esperada_final = ultimo_indice_original - pd.DateOffset(months=12)
    
    # Ajustamos a fin de mes para la comparación exacta
    assert resultado.index[-1] == fecha_esperada_final + pd.offsets.MonthEnd(0)
    
    # 4. Verificar que se mantienen las columnas originales más los dos nuevos targets
    assert 'target_6m' in resultado.columns
    assert 'target_12m' in resultado.columns
    assert 'feature_test' in resultado.columns