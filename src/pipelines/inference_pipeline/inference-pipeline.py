# src/pipelines/inference_pipeline/inference_pipeline.py

import datetime
import os
import sys
from pathlib import Path

import hopsworks
from loguru import logger
from xgboost import XGBRegressor

# Setup path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_PATH = _PROJECT_ROOT / "src"
for path in [str(_PROJECT_ROOT), str(SRC_PATH)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import config  # noqa: E402

settings = config.HopsworksSettings(_env_file=_PROJECT_ROOT / ".env")


def run_inference() -> None:
    logger.info("🔐 Conectando a Hopsworks...")
    if settings.HOPSWORKS_API_KEY is not None:
        os.environ["HOPSWORKS_API_KEY"] = settings.HOPSWORKS_API_KEY.get_secret_value()
    project = hopsworks.login()
    fs = project.get_feature_store()

    logger.info("📦 Recuperando el último modelo registrado")
    mr = project.get_model_registry()

    models = mr.get_models(name="covid_death_predictor_xgboost")

    for model_metadata in reversed(models):
        try:
            model_path = model_metadata.download()
            break  # ✅ Descarga exitosa
        except Exception as e:
            logger.warning(f"❌ No se pudo descargar modelo versión {model_metadata.version}: {e}")
    else:
        raise RuntimeError()

    logger.info(f"📥 Modelo descargado: {model_path}")
    model = XGBRegressor()
    model.load_model(os.path.join(model_path, "model.json"))

    logger.info("🧠 Cargando Feature View")
    fv = model_metadata.get_feature_view()

    # Obtener predicción para los últimos 10 días (ajustable)
    recent_date = datetime.datetime.now() - datetime.timedelta(days=10)
    recent_data = fv.get_batch_data(start_time=recent_date.strftime("%Y-%m-%d"))

    logger.info("🔮 Generando predicciones...")
    X_pred = recent_data.drop(
        columns=[col for col in ["death_count", "date_of_interest"] if col in recent_data.columns]
    )
    predictions = model.predict(X_pred)

    output_df = recent_data[["date_of_interest"]].copy()
    output_df["predicted_death_count"] = predictions.round().astype(int)

    logger.info("📊 Resultado de predicción:")
    logger.info(output_df.tail())

    logger.info("📤 Subiendo predicciones al Feature Store...")
    pred_fg = fs.get_or_create_feature_group(
        name="covid_death_predictions",
        version=1,
        description="Predicciones de muertes por COVID-19 para monitoreo",
        primary_key=["date_of_interest"],
        event_time="date_of_interest",
    )

    pred_fg.insert(output_df, wait=True)
    logger.success("✅ Predicciones almacenadas correctamente")


if __name__ == "__main__":
    run_inference()
