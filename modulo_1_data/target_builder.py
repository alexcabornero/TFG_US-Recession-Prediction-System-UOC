import pandas as pd
import os

class TargetBuilder:
    """Clase para la construcción de las variables objetivo (targets) proyectadas.

    Transforma el indicador contemporáneo USREC en etiquetas predictivas
    con horizontes de 6 y 12 meses.
    """

    def __init__(self, ruta_procesada: str = "data/processed"):
        """Inicializa el constructor del target.

        Args:
            ruta_procesada: Directorio donde residen los datasets procesados.
        """
        self.ruta_procesada = ruta_procesada

    def construir_targets(self, nombre_maestro: str = "dataset_maestro.csv") -> pd.DataFrame:
        """Genera los targets desplazados y los une a las features sin normalizar.

        Args:
            nombre_maestro: Dataset con valores originales (para extraer usrec binario y features).

        Returns:
            DataFrame unificado con features y targets, sin valores nulos, sin normalizar.
        """
        ruta_maestro = os.path.join(self.ruta_procesada, nombre_maestro)

        if not os.path.exists(ruta_maestro):
            raise FileNotFoundError(f"No se encontró {ruta_maestro} en data/processed")

        # 1. Cargar dataset maestro (sin normalizar)
        df_maestro = pd.read_csv(ruta_maestro, index_col=0, parse_dates=True)
        df_features = df_maestro.copy()

        # 2. Extraer USREC original (binario: 0 o 1)
        # Nota: usrec en df_norm está normalizado (centrado), no sirve como target discreto.
        usrec_real = df_maestro['usrec']

        # 3. Crear targets desplazados (Lead targets)
        # shift(-N) mueve los valores del futuro al presente.
        # En el mes t, target_6m tendrá el valor de USREC del mes t+6.
        target_6m = usrec_real.shift(-6)
        target_12m = usrec_real.shift(-12)

        # 4. Unir targets a las features sin normalizar
        # Mantenemos usrec como feature (estado actual de recesión es un predictor válido)
        # La normalización se aplicará dentro del pipeline de CV en baseline.py
        
        df_final = df_features.copy()
        df_final['target_6m'] = target_6m
        df_final['target_12m'] = target_12m

        # 5. Limpieza de NaNs generados por el shift
        # El shift de -12 genera 12 nulos al final del dataset.
        # Estos registros no pueden usarse para entrenamiento supervisado.
        antes = len(df_final)
        df_final = df_final.dropna(subset=['target_6m', 'target_12m'])
        despues = len(df_final)

        print(f"Targets generados. Registros eliminados por horizonte temporal: {antes - despues}")
        
        return df_final

    def guardar_dataset_final(self, df: pd.DataFrame, nombre_archivo: str = "dataset_final.csv") -> str:
        """Guarda el dataset definitivo listo para Machine Learning.

        Args:
            df: DataFrame con features y targets.
            nombre_archivo: Nombre del CSV de salida.

        Returns:
            Ruta completa del archivo guardado.
        """
        ruta_completa = os.path.join(self.ruta_procesada, nombre_archivo)
        df.to_csv(ruta_completa)
        return ruta_completa

if __name__ == "__main__":
    builder = TargetBuilder()
    try:
        print("Iniciando construcción de la variable objetivo...")
        df_listo = builder.construir_targets()
        
        ruta = builder.guardar_dataset_final(df_listo)
        
        print(f"\n¡Éxito! Dataset final generado en: {ruta}")
        print(f"  - Dimensiones: {df_listo.shape}")
        print(f"  - Features: {df_listo.columns[:-2].tolist()}")
        print(f"  - Targets: {df_listo.columns[-2:].tolist()}")
        print(f"  - Periodo útil: {df_listo.index.min().date()} a {df_listo.index.max().date()}")
        
        # Verificar balance de clases en target_6m
        balance = df_listo['target_6m'].value_counts(normalize=True) * 100
        print(f"\nBalance de clases (target_6m):")
        print(f"  - Sin recesión: {balance[0]:.2f}%")
        print(f"  - En recesión: {balance[1]:.2f}%")

    except Exception as e:
        print(f"Error al construir el target: {e}")
