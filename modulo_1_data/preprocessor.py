import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from typing import List, Optional

class DataPreprocessor:
    """Clase para el preprocesamiento, limpieza y alineación de series macroeconómicas.

    Attributes:
        ruta_bruta: Ruta al directorio con los archivos CSV en bruto.
    """

    def __init__(self, ruta_bruta: str = "data/raw"):
        """Inicializa el preprocesador con la ruta de los datos.

        Args:
            ruta_bruta: Directorio que contiene los archivos CSV originales.
        """
        self.ruta_bruta = ruta_bruta

    def load_sp500(self) -> pd.Series:
        """Carga el S&P 500, limpia el formato multi-índice de Yahoo y realiza resampling.

        El archivo sp500.csv descargado en la Tarea 1.2 contiene datos diarios con cabeceras
        complejas. Esta función extrae el cierre ('Close') y lo convierte a mensual.

        Returns:
            Serie con el S&P 500 en frecuencia mensual (último valor del mes).
        """
        ruta_archivo = os.path.join(self.ruta_bruta, "sp500.csv")
        
        # Los CSV de Yahoo Finance suelen tener cabeceras de varias líneas (Price, Ticker, Date)
        # Fila 1: Price, Close...
        # Fila 2: Ticker, ^GSPC...
        # Fila 3: Date, ... (índice)
        # Saltamos las primeras 2 filas para obtener un dataframe limpio.
        df = pd.read_csv(ruta_archivo, skiprows=2, index_col=0, parse_dates=True)
        
        # Ahora las columnas deberían ser las estándar: Close, High, Low, Open, Volume
        if 'Close' in df.columns:
            cierre_sp500 = df['Close']
        else:
            # Alternativa en caso de fallo
            cierre_sp500 = df.iloc[:, 0]

        # Convertir el índice a datetime si no lo es ya
        cierre_sp500.index = pd.to_datetime(cierre_sp500.index)
        
        # Resampling a mensual usando el último valor disponible del mes
        # 'ME' es el alias para Month End (último día del mes)
        # Usamos .last() para obtener el último cierre real del mes bursátil
        sp500_mensual = cierre_sp500.resample('ME').last()
        
        sp500_mensual.name = 'sp500'
        return sp500_mensual

    def load_merged_gold(self) -> pd.Series:
        """Une los datos históricos de Oro (GitHub) con los modernos (Yahoo Finance).

        Utiliza oro_historico.csv para el periodo 1833-1999 y oro.csv para 2000+.

        Returns:
            Serie con el precio del oro unificado y alineado mensualmente.
        """
        ruta_hist = os.path.join(self.ruta_bruta, "oro_historico.csv")
        ruta_moderna = os.path.join(self.ruta_bruta, "oro.csv")
        
        # 1. Cargar histórico (1833-2021 aprox, pero solo usaremos hasta 1999)
        df_hist = pd.read_csv(ruta_hist, index_col=0, parse_dates=True)
        oro_hist = df_hist['Price']
        
        # 2. Cargar moderno (Yahoo, 2000+)
        # Saltamos la cabecera multi-línea de Yahoo
        df_moderna = pd.read_csv(ruta_moderna, skiprows=2, index_col=0, parse_dates=True)
        if 'Close' in df_moderna.columns:
            oro_moderno = df_moderna['Close']
        else:
            oro_moderno = df_moderna.iloc[:, 0]
            
        # 3. Resamplear ambos a frecuencia mensual (ME)
        oro_hist.index = pd.to_datetime(oro_hist.index)
        oro_hist_mensual = oro_hist.resample('ME').last()
        
        oro_moderno.index = pd.to_datetime(oro_moderno.index)
        oro_moderno_mensual = oro_moderno.resample('ME').last()

        # 4. Splicing: Parte histórica estricta (< 2000)
        hist_pre_2000 = oro_hist_mensual[oro_hist_mensual.index < '2000-01-01']
        
        # 5. Parte moderna (>= 2000): Priorizar moderno y rellenar nulos con histórico
        moderno_post_2000 = oro_moderno_mensual[oro_moderno_mensual.index >= '2000-01-01']
        hist_post_2000 = oro_hist_mensual[oro_hist_mensual.index >= '2000-01-01']
        
        # combine_first preserva moderno y rellena sus NaNs con hist_post_2000
        moderno_completado = moderno_post_2000.combine_first(hist_post_2000)
        
        # 6. Unir ambas partes
        oro_unificado = pd.concat([hist_pre_2000, moderno_completado])
        
        oro_unificado.name = 'precio_oro'
        return oro_unificado

    def save_to_processed(self, datos: pd.Series | pd.DataFrame, nombre_archivo: str) -> str:
        """Guarda una serie o dataframe procesado en el directorio de datos procesados.

        Args:
            datos: Datos de pandas a guardar.
            nombre_archivo: Nombre del archivo CSV.

        Returns:
            Ruta completa del archivo guardado.
        """
        ruta_procesada = os.path.join(os.path.dirname(self.ruta_bruta), "processed")
        if not os.path.exists(ruta_procesada):
            os.makedirs(ruta_procesada)
            
        ruta_completa = os.path.join(ruta_procesada, nombre_archivo)
        datos.to_csv(ruta_completa)
        return ruta_completa

    def alinear_todas_las_series(self, fecha_inicio: Optional[str] = None) -> pd.DataFrame:
        """Carga todas las series disponibles y las alinea en un único DataFrame mensual.

        Incluye las 12 series de FRED, el S&P 500 y el Oro.

        Args:
            fecha_inicio: Fecha mínima para el índice temporal.

        Returns:
            DataFrame con todas las series alineadas y frecuencia 'ME'.
        """
        series_fred = [
            "usrec", "unrate", "cpiaucsl", "indpro", "m2sl", "wtisplc",
            "icsa", "houst", "gs10", "tb3ms", "baa", "ppiaco",
            "usalolitoaastsam" 
        ]
        
        dicc_series = {}

        # 1. Cargar series FRED
        for id_serie in series_fred:
            ruta = os.path.join(self.ruta_bruta, f"{id_serie}.csv")
            if os.path.exists(ruta):
                df = pd.read_csv(ruta, index_col=0, parse_dates=True)
                nombre_col = df.columns[0]
                serie = df[nombre_col]
                # Asegurar frecuencia mensual al final del mes
                # Esto maneja series diarias (sp500), semanales (icsa) o mensuales (unrate)
                serie_mensual = serie.resample('ME').last()
                dicc_series[id_serie] = serie_mensual

        # 2. Cargar S&P 500 (ya viene normalizado por load_sp500)
        dicc_series['sp500'] = self.load_sp500()

        # 3. Cargar Oro (ya viene normalizado por load_merged_gold)
        dicc_series['precio_oro'] = self.load_merged_gold()

        # 4. Crear DataFrame base
        df_final = pd.DataFrame(dicc_series)

        # 5. Asegurar frecuencia mensual 'ME' y rango dinámico
        # En lugar de 1967 fijo, buscamos el primer punto donde TODAS las series coinciden
        
        # Eliminar filas iniciales donde faltan series
        # Buscamos la primera fila que no tenga ningún NaN
        df_final = df_final.dropna(how='any', axis=0)
        
        if fecha_inicio:
            df_final = df_final[df_final.index >= fecha_inicio]
        
        # Reindexar para asegurar que no falten meses en el índice (frecuencia estricta)
        if not df_final.empty:
            indice_maestro = pd.date_range(
                start=df_final.index.min(), 
                end=df_final.index.max(), 
                freq='ME'
            )
            df_final = df_final.reindex(indice_maestro)

        return df_final

    def generar_reporte_nulos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Genera un reporte detallado de valores faltantes y calidad de datos.

        Args:
            df: DataFrame alineado a analizar.

        Returns:
            DataFrame con estadísticas de nulos por columna.
        """
        reporte = pd.DataFrame({
            'nulos': df.isnull().sum(),
            'porcentaje': (df.isnull().sum() / len(df)) * 100,
            'primera_fecha_valida': df.apply(lambda x: x.first_valid_index()),
            'ultima_fecha_valida': df.apply(lambda x: x.last_valid_index())
        })
        
        # Ordenar por número de nulos descendente
        reporte = reporte.sort_values(by='nulos', ascending=False)
        return reporte

    def limpiar_y_completar_datos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica la estrategia de imputación y limpieza final.

        Estrategia:
        1. Interpolación lineal para huecos intermedios (si los hubiera).
        2. Truncamiento de nulos al final (donde faltan publicaciones oficiales).

        Args:
            df: DataFrame alineado a limpiar.

        Returns:
            DataFrame limpio y sin nulos.
        """
        # 1. Tratar huecos intermedios (si los hubiera) con interpolación lineal
        df_imputado = df.interpolate(method='linear', limit_area='inside')
        
        # 2. Eliminar filas con nulos (principalmente al final por retardo de publicación)
        # Esto asegura un dataset 100% denso para el modelado
        df_limpio = df_imputado.dropna()
        
        return df_limpio

    def normalizar_datos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estandariza los datos a media 0 y desviación estándar 1.

        Args:
            df: DataFrame limpio a normalizar.

        Returns:
            DataFrame normalizado con las mismas columnas e índice.
        """
        escalador = StandardScaler()
        
        # Ajustar y transformar (ignora el índice automáticamente)
        datos_escalados = escalador.fit_transform(df)
        
        # Reconstruir el DataFrame con índice y columnas originales
        df_normalizado = pd.DataFrame(
            datos_escalados, 
            index=df.index, 
            columns=df.columns
        )
        
        return df_normalizado

if __name__ == "__main__":
    # Prueba rápida de ejecución y guardado
    preprocesador = DataPreprocessor()
    try:
        # 1. Procesar S&P 500
        sp500 = preprocesador.load_sp500()
        ruta_sp500 = preprocesador.save_to_processed(sp500, "sp500_mensual.csv")
        print(f"S&P 500 guardado en: {ruta_sp500}")
        print(f"  - Meses: {len(sp500)}. Rango: {sp500.index.min()} a {sp500.index.max()}")
        
        # 2. Procesar Oro
        oro = preprocesador.load_merged_gold()
        ruta_oro = preprocesador.save_to_processed(oro, "oro_unificado.csv")
        print(f"Oro unificado guardado en: {ruta_oro}")
        print(f"  - Meses: {len(oro)}. Rango: {oro.index.min()} a {oro.index.max()}")

        # 3. Alinear todas las series
        print("\nIniciando alineación temporal de todas las series...")
        df_alineado = preprocesador.alinear_todas_las_series()
        ruta_final = preprocesador.save_to_processed(df_alineado, "dataset_alineado.csv")
        print(f"Dataset alineado guardado en: {ruta_final}")
        print(f"  - Dimensiones: {df_alineado.shape} (Meses, Indicadores)")
        print(f"  - Rango: {df_alineado.index.min()} a {df_alineado.index.max()}")

        # 4. Reporte de nulos inicial (EDA 1.3.3)
        print("\n--- Reporte de Calidad Inicial (EDA) ---")
        reporte_inicial = preprocesador.generar_reporte_nulos(df_alineado)
        print(reporte_inicial)

        # 5. Imputación y Limpieza Final (1.3.4)
        print("\n--- Aplicando Limpieza e Imputación ---")
        df_final = preprocesador.limpiar_y_completar_datos(df_alineado)
        
        # Guardar dataset alineado intermedio
        preprocesador.save_to_processed(df_final, "dataset_alineado.csv")
        
        # 6. Cargar Dataset Maestro (Con spreads calculados)
        print("\n--- Cargando Dataset Maestro (con spreads) ---")
        ruta_maestra_csv = os.path.join("data/processed", "dataset_maestro.csv")
        if os.path.exists(ruta_maestra_csv):
            df_maestro = pd.read_csv(ruta_maestra_csv, index_col=0, parse_dates=True)
            print(f"Dataset maestro cargado. Dimensiones: {df_maestro.shape}")
        else:
            print("Aviso: No se encontró dataset_maestro.csv. Usando dataset_alineado.csv")
            df_maestro = df_final

        # 7. Normalización Final (Hito 1 corregido)
        print("\n--- Normalizando Datos (StandardScaler) ---")
        df_normalizado = preprocesador.normalizar_datos(df_maestro)
        
        # Guardar dataset normalizado final
        ruta_normalizada = preprocesador.save_to_processed(df_normalizado, "dataset_normalizado.csv")
        print(f"Dataset normalizado final guardado en: {ruta_normalizada}")
        print(f"  - Dimensiones: {df_normalizado.shape}")
        
        # Verificar estadísticas rápidas
        print("\nEstadísticas del dataset normalizado (primeras 5 columnas):")
        print(df_normalizado.iloc[:, :5].describe().loc[['mean', 'std']])
        
        print("\n¡Proceso de preprocesamiento y normalización finalizado con éxito!")
        
    except Exception as e:
        print(f"Error en el preprocesamiento: {e}")
        import traceback
        traceback.print_exc()
