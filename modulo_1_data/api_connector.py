"""Módulo para la conexión automatizada a las APIs de FRED y Yahoo Finance.

Este módulo centraliza la lógica de descarga de datos macroeconómicos y de mercado,
gestionando la autenticación en FRED.
"""

import os
import time
from typing import Optional

import pandas as pd
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()


class FREDConnector:
    """Conector para la API de Federal Reserve Economic Data (FRED)."""

    def __init__(self, clave_api: Optional[str] = None):
        """Inicializa el conector de FRED.

        Args:
            clave_api: Clave de API de FRED. Si no se proporciona, se busca en la variable
                de entorno FRED_API_KEY.

        Raises:
            ValueError: Si no se encuentra una clave de API válida.
        """
        self.clave_api = clave_api or os.getenv("FRED_API_KEY")
        if not self.clave_api:
            raise ValueError(
                "No se encontró la clave de API de FRED. "
                "Asegúrate de configurar FRED_API_KEY en tu archivo .env"
            )
        self.cliente = Fred(api_key=self.clave_api)

    def obtener_serie(self, id_serie: str, **kwargs) -> pd.Series:
        """Descarga una serie temporal de FRED.

        Args:
            id_serie: Identificador de la serie en FRED (ej. 'USREC').
            **kwargs: Parámetros adicionales para la API de FRED (ej. observation_start).

        Returns:
            Serie temporal descargada.

        Raises:
            Exception: Si ocurre un error en la conexión o descarga.
        """
        try:
            serie = self.cliente.get_series(id_serie, **kwargs)
            if serie.empty:
                raise ValueError(f"La serie '{id_serie}' no contiene datos.")
            return serie
        except Exception as e:
            # Implementar lógica sencilla de reintento si es necesario
            print(f"Error al descargar la serie '{id_serie}' de FRED: {e}")
            raise


class YahooFinanceConnector:
    """Conector para la API de Yahoo Finance."""

    @staticmethod
    def obtener_datos(ticker: str, inicio: str, fin: str, intervalo: str = "1mo") -> pd.DataFrame:
        """Descarga datos históricos de Yahoo Finance.

        Args:
            ticker: Ticker del activo (ej. '^GSPC').
            inicio: Fecha de inicio (YYYY-MM-DD).
            fin: Fecha de fin (YYYY-MM-DD).
            intervalo: Intervalo de los datos (defecto: '1mo').

        Returns:
            DataFrame con los datos históricos.

        Raises:
            Exception: Si ocurre un error en la descarga.
        """
        try:
            datos = yf.download(ticker, start=inicio, end=fin, interval=intervalo, progress=False)
            if datos.empty:
                raise ValueError(f"No se encontraron datos para el ticker '{ticker}'.")
            return datos
        except Exception as e:
            print(f"Error al descargar datos de Yahoo Finance para '{ticker}': {e}")
            raise


if __name__ == "__main__":
    # Prueba rápida de conexión
    try:
        print("Probando conexión a FRED...")
        fred = FREDConnector()
        usrec = fred.obtener_serie("USREC", observation_start="2024-01-01")
        print(f"FRED OK: Descargados {len(usrec)} puntos de USREC.")

        print("\nProbando conexión a Yahoo Finance...")
        yf_conn = YahooFinanceConnector()
        sp500 = yf_conn.obtener_datos("^GSPC", inicio="2024-01-01", fin="2024-03-01")
        print(f"Yahoo Finance OK: Descargadas {len(sp500)} filas de S&P 500.")

    except Exception as exc:
        print(f"\nERROR durante la prueba de conexión: {exc}")
