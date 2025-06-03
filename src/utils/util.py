from typing import Any

import pandas as pd
import requests  # type: ignore
from loguru import logger


class NYCOpenDataDownloadError(Exception):  # type: ignore[misc]
    """Excepción personalizada para errores de descarga de datos de NYC Open Data."""

    default_message = "Error en la descarga de datos"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.default_message)


def get_nyc_covid_data(dataset_id: str, app_token: str, limit: int = 2000) -> pd.DataFrame:
    """Descarga datos de COVID-19 desde el portal de NYC OpenData.

    Args:
        dataset_id (str): ID del conjunto de datos (ej. 'rc75-m7u3').
        app_token (str): Token de la aplicación registrado en NYC Open Data.
        limit (int): Límite de filas a descargar.

    Returns:
        pd.DataFrame: DataFrame con los datos descargados.
    """
    HTTP_OK = 200
    url = f"https://data.cityofnewyork.us/resource/{dataset_id}.json?$limit={limit}"
    headers = {"X-App-Token": app_token}
    response = requests.get(url, headers=headers, timeout=10)

    if response.status_code != HTTP_OK:
        logger.error(f"Error al obtener los datos: {response.status_code}")
        raise NYCOpenDataDownloadError()

    data = response.json()
    return pd.DataFrame(data)


class MissingColumnsError(ValueError):  # type: ignore[misc]
    """Faltan columnas obligatorias en el DataFrame."""

    default_message = "Faltan columnas obligatorias"

    def __init__(self, missing: set[str] | None = None) -> None:
        message = f"{self.default_message}: {missing}" if missing else self.default_message
        super().__init__(message)


def validate_columns(df: pd.DataFrame, expected_columns: list[str]) -> None:
    """Valida que existan las columnas esperadas en el DataFrame.

    Args:
        df (pd.DataFrame): DataFrame a validar.
        expected_columns (List[str]): Lista de nombres de columnas requeridas.

    Raises:
        MissingColumnsError: Si alguna columna no está presente.
    """
    missing = set(expected_columns) - set(df.columns)
    if missing:
        raise MissingColumnsError(missing)


def convert_types(df: pd.DataFrame, column_types: dict[str, Any]) -> pd.DataFrame:
    """Convierte los tipos de columnas según especificado.

    Args:
        df (pd.DataFrame): DataFrame a convertir.
        column_types (Dict[str, str]): Diccionario columna -> tipo (ej. 'fecha': 'datetime64[ns]').

    Returns:
        pd.DataFrame: DataFrame con tipos convertidos.
    """
    for col, typ in column_types.items():
        try:
            df[col] = df[col].astype(typ)
        except Exception as e:
            logger.warning(f"No se pudo convertir la columna {col} a tipo {typ}: {e}")
    return df


def filter_by_date(df: pd.DataFrame, date_col: str, min_date: str) -> pd.DataFrame:
    """Filtra el DataFrame por fecha mínima.

    Args:
        df (pd.DataFrame): DataFrame a filtrar.
        date_col (str): Nombre de la columna de fecha.
        min_date (str): Fecha mínima en formato 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame: DataFrame filtrado.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    return df[df[date_col] >= pd.to_datetime(min_date)]
