from pathlib import Path

import pandas as pd

from src.pipelines.training_pipeline.utils import validate_train_test_split


# Caso válido: misma distribución
def test_validate_train_test_split_pass() -> None:
    data = {
        "feature1": list(range(100)),
        "feature2": list(range(100, 200)),
    }
    y = pd.Series([i % 2 for i in range(100)], name="target")
    df = pd.DataFrame(data)
    df["target"] = y

    # Mezclar aleatoriamente
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Separar
    train_df = df.iloc[:70]
    test_df = df.iloc[70:]

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    # Ahora sí: debería pasar
    validate_train_test_split(X_train, X_test, y_train, y_test, label="target")


# Caso inválido: distribución de label totalmente distinta
def test_validate_train_test_split_fail() -> None:
    X_train = pd.DataFrame(
        {
            "feature1": list(range(50)),
            "feature2": list(range(100, 150)),
        }
    )
    y_train = pd.Series([0] * 50, name="target")

    X_test = pd.DataFrame(
        {
            "feature1": list(range(50, 100)),
            "feature2": list(range(150, 200)),
        }
    )
    y_test = pd.Series([1] * 50, name="target")

    # No debe lanzar error, solo generar el archivo HTML
    validate_train_test_split(X_train, X_test, y_train, y_test, label="target")

    # Verificar que se genera el archivo
    expected_report = Path("model/train_test_validation_report.html")
    assert expected_report.exists(), f"No se generó el reporte: {expected_report}"
