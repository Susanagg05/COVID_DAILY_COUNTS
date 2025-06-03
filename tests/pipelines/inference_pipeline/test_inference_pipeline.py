import pandas as pd
import numpy as np
from xgboost import XGBRegressor

def test_inference_with_mock_data():
    # Crear modelo dummy
    model = XGBRegressor()
    X_train = pd.DataFrame({
        "feature1": np.random.rand(100),
        "feature2": np.random.rand(100),
    })
    y_train = np.random.randint(0, 10, 100)
    model.fit(X_train, y_train)

    # Crear datos de entrada simulados
    X_test = pd.DataFrame({
        "feature1": np.random.rand(5),
        "feature2": np.random.rand(5),
    })
    dates = pd.date_range(start="2025-06-01", periods=5)

    # Hacer predicción
    y_pred = model.predict(X_test)

    # Ensamblar DataFrame de salida
    output_df = pd.DataFrame({
        "date_of_interest": dates,
        "predicted_death_count": np.round(y_pred).astype(int)
    })

    # Verificar forma y columnas
    assert output_df.shape == (5, 2)
    assert "date_of_interest" in output_df.columns
    assert "predicted_death_count" in output_df.columns
    assert output_df["predicted_death_count"].dtype == int