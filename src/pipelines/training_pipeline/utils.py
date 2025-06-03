# src/pipelines/training_pipeline/utils.py
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import train_test_validation
from loguru import logger
from pathlib import Path
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

def validate_train_test_split(X_train: pd.DataFrame, X_test: pd.DataFrame,
                               y_train: pd.Series, y_test: pd.Series,
                               label: str = "death_count") -> None:
    """Valida la separación Train/Test para detectar fugas de datos o distribución anómala"""

    logger.info("🔍 Validando la separación Train/Test con Deepchecks...")

    # Construir datasets para deepchecks
    train_data = X_train.copy()
    train_data[label] = y_train.values

    test_data = X_test.copy()
    test_data[label] = y_test.values

    train_ds = Dataset(train_data, label=label)
    test_ds = Dataset(test_data, label=label)

    # Ejecutar validación
    suite = train_test_validation()
    result = suite.run(train_ds, test_ds)

    if result.passed():
        logger.success("✅ Validación de Train/Test completada sin problemas.")
    else:
        logger.warning("⚠️ Se encontraron advertencias en la separación Train/Test.")
        report_path = _PROJECT_ROOT / "model" / "train_test_validation_report.html"
        report_path.parent.mkdir(parents=True, exist_ok=True)  # 👈 Crea la carpeta si no existe
        result.save_as_html(str(report_path))
        raise ValueError(f"❌ Validación Train/Test fallida. Revisa el reporte generado en {report_path}")
