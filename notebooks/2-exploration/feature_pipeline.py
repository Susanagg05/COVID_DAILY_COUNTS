"""Feature pipeline for the Hopsworks COVID-19 Reporting project"""

import datetime
import os  # For path manipulation
import sys  # For sys.path modification
import warnings

import hopsworks
import pandas as pd
from loguru import logger

# Add the 'src' directory to sys.path
# This allows modules in 'src' (like config.py) and packages in 'src' (like utils)
# to be imported directly when this script is run.
# __file__ is src/pipelines/feature_pipeline/feature-pipeline.py
# os.path.dirname(__file__) is src/pipelines/feature_pipeline
# os.path.join(os.path.dirname(__file__), '..', '..') is src/
_SRCDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRCDIR not in sys.path:
    sys.path.insert(0, _SRCDIR)

from src import config  # noqa: E402
from src.utils import util  # noqa: E402

# Cargar variables del entorno
settings = config.HopsworksSettings(_env_file=f"{_SRCDIR}/.env")
warnings.filterwarnings("ignore")


class MissingEnvVarError(ValueError):
    """Falta una variable de entorno obligatoria."""

    def __init__(self, var_name: str) -> None:
        super().__init__(f"'{var_name}' no está definido en el .env")


# Validar que las variables requeridas no sean None
if settings.HOPSWORKS_API_KEY is None:
    raise MissingEnvVarError("HOPSWORKS_API_KEY")
if settings.HOPSWORKS_HOST is None:
    raise MissingEnvVarError("HOPSWORKS_HOST")
if settings.NYC_APP_TOKEN is None:
    raise MissingEnvVarError("NYC_APP_TOKEN")

logger.info("🔐 Setting up Hopsworks connection...")
print("HOST:", settings.HOPSWORKS_HOST)

project = hopsworks.login(
    api_key_value=settings.HOPSWORKS_API_KEY.get_secret_value(),
    project=settings.HOPSWORKS_PROJECT,
    host=settings.HOPSWORKS_HOST.removeprefix("https://"),
)


logger.info("🔗 Conexión a Hopsworks establecida")

fs = project.get_feature_store()

logger.info("📅 Definir fecha de procesamiento")
today = datetime.date.today()

logger.info("📦 Recuperar Feature Group de COVID-19")
covid_fg = fs.get_feature_group(
    name="covid_daily_counts",
    version=1,
)

logger.info("⬇️ Descargar datos de NYC Open Data")
covid_df = util.get_nyc_covid_data(
    dataset_id=settings.NYC_DATASET_ID,
    app_token=settings.NYC_APP_TOKEN.get_secret_value(),
    limit=settings.NYC_DEFAULT_LIMIT,
)

logger.info("🔍 Obtener esquema del Feature Group")
feature_schema = covid_fg.schema
expected_columns = [feature.name for feature in feature_schema]

logger.info("🛠️ Corrigiendo tipos de datos")
# Asegurar que 'date_of_interest' sea datetime
if "date_of_interest" in covid_df.columns:
    covid_df["date_of_interest"] = pd.to_datetime(covid_df["date_of_interest"], errors="coerce")

# Convertir columnas numéricas si están en el esquema
for col in expected_columns:
    if col in covid_df.columns and col != "date_of_interest":
        covid_df[col] = pd.to_numeric(covid_df[col], errors="coerce")

# Filtrar solo las columnas que existen en el FG
covid_df = covid_df[expected_columns]

logger.debug(f"✅ Shape: {covid_df.shape}")
logger.debug(f"🧬 Dtypes: {covid_df.dtypes}")


covid_df.rename(
    columns={
        "case_count": "CASE_COUNT",
        "probable_case_count": "PROBABLE_CASE_COUNT",
        "hospitalized_count": "HOSPITALIZED_COUNT",
        "death_count": "DEATH_COUNT",
        "case_count_7day_avg": "CASE_COUNT_7DAY_AVG",
        "all_case_count_7day_avg": "ALL_CASE_COUNT_7DAY_AVG",
        "hosp_count_7day_avg": "HOSP_COUNT_7DAY_AVG",
        "death_count_7day_avg": "DEATH_COUNT_7DAY_AVG",
        "date_of_interest": "date_of_interest",
    },
    inplace=True,
)

# (Opcional) Validación con Pandera
try:
    from src.validations.covid_schema import covid_schema


    covid_df = covid_schema.validate(covid_df)
    logger.info("✔️ Validación completada")
except ImportError:
    logger.warning("⚠️ No se encontró un esquema de validación. Continuando sin validación.")

logger.info("📤 Insertar datos en Feature Store")
covid_fg.insert(covid_df, wait=True)

logger.info("✅ COVID-19 data inserted successfully")
