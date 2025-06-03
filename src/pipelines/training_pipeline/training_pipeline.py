# ruff: noqa: E402, PLR0915

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import hopsworks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import model_evaluation
from loguru import logger
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from xgboost import XGBRegressor, plot_importance

# Ruta raíz del proyecto
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_PATH = _PROJECT_ROOT / "src"

for path in [str(_PROJECT_ROOT), str(SRC_PATH)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import config
from src.pipelines.training_pipeline.utils import validate_train_test_split


def run_training() -> None:
    settings = config.HopsworksSettings(_env_file=_PROJECT_ROOT / ".env")
    warnings.filterwarnings("ignore")

    logger.info("🔐 Conectando a Hopsworks...")
    if settings.HOPSWORKS_API_KEY is not None:
        api_key = settings.HOPSWORKS_API_KEY.get_secret_value()
        os.environ["HOPSWORKS_API_KEY"] = api_key
    project = hopsworks.login()
    fs = project.get_feature_store()

    logger.info("📦 Recuperando Feature Group: covid_daily_counts")
    covid_fg = fs.get_feature_group(
        name="covid_daily_counts",
        version=1,
    )

    logger.info("🧠 Seleccionando features para entrenamiento")
    # Selección de columnas relevantes
    selected_features = covid_fg.select(
        [
            "case_count",
            "probable_case_count",
            "hospitalized_count",
            "case_count_7day_avg",
            "all_case_count_7day_avg",
            "hosp_count_7day_avg",
            "death_count_7day_avg",
            "date_of_interest",
            "death_count",  # Etiqueta
        ]
    )

    logger.info("📚 Creando Feature View")
    feature_view = fs.get_or_create_feature_view(
        name="covid_death_fv",
        version=1,
        description="Predicción de muertes por COVID-19 a partir de variables epidemiológicas",
        labels=["death_count"],
        query=selected_features,
    )

    logger.info("📆 Separando conjunto de entrenamiento y prueba")

    test_start = datetime.strptime("2022-02-01", "%Y-%m-%d")
    X_train, X_test, y_train, y_test = feature_view.train_test_split(test_start=test_start)

    validate_train_test_split(X_train, X_test, y_train, y_test)

    logger.info("🧼 Preprocesando: eliminando columnas clave")
    dates = X_test["date_of_interest"].copy()
    X_train = X_train.drop(["date_of_interest"], axis=1)
    X_test = X_test.drop(["date_of_interest"], axis=1)

    # 🔁 Validación cruzada con TimeSeriesSplit
    logger.info("🔁 Ejecutando validación cruzada con TimeSeriesSplit...")
    model = XGBRegressor()
    tscv = TimeSeriesSplit(n_splits=5)
    mae_scorer = make_scorer(mean_absolute_error)

    cv_scores = cross_val_score(model, X_train, y_train.values.ravel(), cv=tscv, scoring=mae_scorer)
    logger.info(f"📉 MAE en CV: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

    # ==================== Entrenamiento final ====================
    logger.info("🏋️ Entrenando modelo XGBoost")
    model.fit(X_train, y_train)

    logger.info("🔮 Generando predicciones")
    y_pred = model.predict(X_test)

    # ==================== Evaluación ====================
    mae = mean_absolute_error(y_test.iloc[:, 0], y_pred)
    mse = mean_squared_error(y_test.iloc[:, 0], y_pred)
    r2 = r2_score(y_test.iloc[:, 0], y_pred)

    logger.success(f"MAE: {mae:.2f}")
    logger.success(f"MSE: {mse:.2f}")
    logger.success(f"R²: {r2:.4f}")

    if mae > cv_scores.mean() * 1.2:
        logger.warning("⚠️ Posible underfitting detectado (test MAE mucho mayor que CV).")
    elif mae < cv_scores.mean() * 0.8:
        logger.warning("⚠️ Posible overfitting detectado (test MAE mucho menor que CV).")
    else:
        logger.info("✅ Comportamiento generalizado entre CV y Test.")

    # ==================== Dataframe final ====================
    logger.info("🧾 Construyendo dataframe final de resultados")
    df = y_test.copy()
    df["predicted_deaths"] = np.round(y_pred).astype(int)
    df["date_of_interest"] = dates
    df = df.sort_values(by="date_of_interest")

    # ==================== Reporte Deepchecks ====================
    logger.info("🧪 Ejecutando evaluación del modelo con Deepchecks")
    train_ds = Dataset(pd.concat([X_train, y_train], axis=1), label="death_count")
    test_ds = Dataset(pd.concat([X_test, y_test], axis=1), label="death_count")

    suite = model_evaluation()
    result = suite.run(train_ds, test_ds, model)

    model_dir = _PROJECT_ROOT / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    result.save_as_html(str(model_dir / "model_evaluation_report.html"))
    logger.info(
        f"📄 Reporte de evaluación guardado en {model_dir / 'model_evaluation_report.html'}"
    )

    # ==================== Visualización ====================
    logger.info("🎨 Exportando visualizaciones")
    images_dir = model_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    plot_importance(model, max_num_features=6)
    plt.tight_layout()
    plt.savefig(images_dir / "feature_importance.png")

    # ==================== Guardado del modelo ====================
    logger.info("💾 Guardando modelo y métricas")
    model.save_model(model_dir / "model.json")

    metrics = {
        "MAE": f"{mae:.2f}",
        "MSE": f"{mse:.2f}",
        "R2": f"{r2:.4f}",
        "CV_MAE_mean": f"{cv_scores.mean():.2f}",
        "CV_MAE_std": f"{cv_scores.std():.2f}",
    }

    mr = project.get_model_registry()
    covid_model = mr.python.create_model(
        name="covid_death_predictor_xgboost",
        metrics=metrics,
        feature_view=feature_view,
        description="Predicción de muertes por COVID-19 en NYC con XGBoost",
    )
    covid_model.save(str(model_dir))

    logger.success("✅ Entrenamiento y registro del modelo completado exitosamente.")


if __name__ == "__main__":
    run_training()
