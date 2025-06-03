import pandas as pd
import pytest
from src.pipelines.training_pipeline.utils import validate_train_test_split

# Caso válido: misma distribución
def test_validate_train_test_split_pass():
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
def test_validate_train_test_split_fail():
    X_train = pd.DataFrame({
        "feature1": list(range(50)),
        "feature2": list(range(100, 150)),
    })
    y_train = pd.Series([0] * 50)

    X_test = pd.DataFrame({
        "feature1": list(range(50, 100)),
        "feature2": list(range(150, 200)),
    })
    y_test = pd.Series([1] * 50)

    with pytest.raises(ValueError, match="Validación Train/Test fallida"):
        validate_train_test_split(X_train, X_test, y_train, y_test, label="target")
