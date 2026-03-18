"""Módulo para la descarga y almacenamiento de series macroeconómicas y de mercado.

Este módulo utiliza los conectores de FRED y Yahoo Finance para descargar los
indicadores definidos en las especificaciones del proyecto.
"""

import os
import pandas as pd
from typing import List, Dict
from modulo_1_data.api_connector import FREDConnector, YahooFinanceConnector

# Configuración de rutas
RUTA_DATOS_BRUTOS = os.path.join("data", "raw")

# Lista de series FRED (Tarea 1.2.1)
SERIES_FRED = {
    "USREC": "NBER Recession Indicator",
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "Consumer Price Index",
    "INDPRO": "Industrial Production Index",
    "M2SL": "M2 Money Stock",
    "WTISPLC": "WTI Crude Oil Price",
    "ICSA": "Initial Cliams (Weekly to Monthly)",
    "HOUST": "Housing Starts",
    "GS10": "10-Year Treasury Yield",
    "TB3MS": "3-Month Treasury Bill",
    "BAA": "Moody's Seasoned Baa Corporate Bond Yield",
    "PPIACO": "Producer Price Index (All Commodities)"
}

# Lista de tickers Yahoo Finance
TICKERS_YF = {
    "^GSPC": "sp500",
    "GC=F": "oro"
}


class DataDownloader:
    """Clase para gestionar la descarga masiva de series."""

    def __init__(self):
        """Inicializa los conectores."""
        self.con_fred = FREDConnector()
        self.con_yf = YahooFinanceConnector()
        
        # Asegurar que el directorio de datos existe
        os.makedirs(RUTA_DATOS_BRUTOS, exist_ok=True)

    def download_fred_series(self, fecha_inicio: str = None) -> None:
        """Descarga todas las series de FRED definidas.

        Args:
            fecha_inicio: Fecha de inicio de la descarga.
        """
        print(f"Iniciando descarga de {len(SERIES_FRED)} series desde FRED...")
        for id_serie, nombre in SERIES_FRED.items():
            try:
                print(f"  Descargando {id_serie} ({nombre})...")
                serie = self.con_fred.obtener_serie(id_serie, observation_start=fecha_inicio)
                
                # Convertir a DataFrame para guardar
                df = serie.to_frame(name="valor")
                ruta_archivo = os.path.join(RUTA_DATOS_BRUTOS, f"{id_serie.lower()}.csv")
                df.to_csv(ruta_archivo)
                print(f"    Guardado en: {ruta_archivo}")
            except Exception as e:
                print(f"    [ERROR] No se pudo descargar {id_serie}: {e}")

    def download_market_data(self, fecha_inicio: str = "1927-01-01", fecha_fin: str = "2026-02-28") -> None:
        """Descarga datos de mercado desde Yahoo Finance.

        Args:
            fecha_inicio: Fecha de inicio.
            fecha_fin: Fecha de fin.
        """
        print(f"Iniciando descarga de {len(TICKERS_YF)} tickers desde Yahoo Finance...")
        for ticker, nombre_archivo in TICKERS_YF.items():
            try:
                print(f"  Descargando {ticker}...")
                # EL S&P 500 se baja en DIARIO para tener histórico completo (desde 1967)
                intervalo = "1d" if ticker == "^GSPC" else "1mo"
                
                datos = self.con_yf.obtener_datos(ticker, inicio=fecha_inicio, fin=fecha_fin, intervalo=intervalo)
                
                ruta_archivo = os.path.join(RUTA_DATOS_BRUTOS, f"{nombre_archivo}.csv")
                datos.to_csv(ruta_archivo)
                print(f"    Guardado en: {ruta_archivo} (Intervalo: {intervalo})")
            except Exception as e:
                print(f"    [ERROR] No se pudo descargar {ticker}: {e}")

    def download_historical_gold(self) -> None:
        """Descarga precios históricos del oro desde un repositorio de GitHub (1833-presente).

        Esta fuente suplementa el histórico de Yahoo Finance para el periodo 1967-2000.
        """
        url = "https://raw.githubusercontent.com/datasets/gold-prices/master/data/monthly.csv"
        print(f"Iniciando descarga de precios históricos del oro desde GitHub...")
        try:
            df = pd.read_csv(url)
            ruta_archivo = os.path.join(RUTA_DATOS_BRUTOS, "oro_historico.csv")
            df.to_csv(ruta_archivo, index=False)
            print(f"    Guardado en: {ruta_archivo}")
        except Exception as e:
            print(f"    [ERROR] No se pudo descargar el oro histórico: {e}")

    def run_full_download(self) -> None:
        """Ejecuta la descarga completa de todos los indicadores."""
        self.download_fred_series()
        self.download_market_data()
        self.download_historical_gold()
        print("\nDescarga completa finalizada.")


if __name__ == "__main__":
    descargador = DataDownloader()
    descargador.run_full_download()
