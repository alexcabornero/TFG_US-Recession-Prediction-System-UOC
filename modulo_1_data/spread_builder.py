import pandas as pd
import os

class SpreadBuilder:
    """Clase para construir indicadores derivados (spreads) a partir de series base.
    """

    def __init__(self, ruta_procesada: str = "data/processed"):
        """Inicializa el constructor de spreads.

        Args:
            ruta_procesada: Directorio donde se encuentran los datos alineados.
        """
        self.ruta_procesada = ruta_procesada

    def cargar_datos_alineados(self) -> pd.DataFrame:
        """Carga el dataset alineado generado en la fase anterior.

        Returns:
            DataFrame con las series temporales alineadas.
        """
        ruta_archivo = os.path.join(self.ruta_procesada, "dataset_alineado.csv")
        if not os.path.exists(ruta_archivo):
            raise FileNotFoundError(f"No se encuentra el archivo: {ruta_archivo}")
        
        return pd.read_csv(ruta_archivo, index_col=0, parse_dates=True)

    def calcular_spreads(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula los diferenciales de tipos y de crédito.

        Args:
            df: DataFrame con las series base (gs10, tb3ms, baa).

        Returns:
            DataFrame original con las nuevas columnas de spreads.
        """
        df_spreads = df.copy()

        # 1. Yield Spread (Diferencial de tipos): Largo plazo - Corto Plazo
        # Refleja la pendiente de la curva de tipos. Inversión = Señal de recesión.
        if 'gs10' in df.columns and 'tb3ms' in df.columns:
            df_spreads['yield_spread'] = df['gs10'] - df['tb3ms']
            print("Spread de tipos (gs10 - tb3ms) calculado.")
        
        # 2. Credit Spread (Diferencial de crédito): Riesgo Corporativo - Deuda Soberana
        # Refleja el aumento del riesgo percibido en el sector privado.
        if 'baa' in df.columns and 'gs10' in df.columns:
            df_spreads['credit_spread'] = df['baa'] - df['gs10']
            print("Spread de crédito (baa - gs10) calculado.")

        return df_spreads

    def guardar_dataset_maestro(self, df: pd.DataFrame) -> str:
        """Guarda el dataset con spreads como el dataset maestro final.

        Args:
            df: DataFrame completo con 16 columnas.

        Returns:
            Ruta del archivo guardado.
        """
        ruta_maestra = os.path.join(self.ruta_procesada, "dataset_maestro.csv")
        df.to_csv(ruta_maestra)
        return ruta_maestra

if __name__ == "__main__":
    builder = SpreadBuilder()
    try:
        print("Cargando datos alineados...")
        df_base = builder.cargar_datos_alineados()
        
        print("Calculando spreads...")
        df_con_spreads = builder.calcular_spreads(df_base)
        
        print("Guardando dataset maestro...")
        ruta = builder.guardar_dataset_maestro(df_con_spreads)
        
        print(f"¡Éxito! Dataset maestro generado en: {ruta}")
        print(f"  - Columnas totales: {len(df_con_spreads.columns)}")
        print(f"  - Nuevas columnas: yield_spread, credit_spread")
        
    except Exception as e:
        print(f"Error al construir spreads: {e}")
