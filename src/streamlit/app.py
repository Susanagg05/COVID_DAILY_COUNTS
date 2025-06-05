# streamlit_app.py

import pandas as pd
from xgboost import XGBRegressor

import streamlit as st

HIGH_DEATH_THRESHOLD = 50
MEDIUM_DEATH_THRESHOLD = 20


@st.cache_resource
def load_model() -> XGBRegressor:
    model = XGBRegressor()
    model.load_model("src/model/model.json")
    return model


def user_input_form() -> pd.DataFrame:
    st.header("📝 Parámetros de entrada")

    col1, col2 = st.columns(2)

    with col1:
        case_count = st.number_input("🔢 Número de casos confirmados", min_value=0, value=200)
        probable_case_count = st.number_input("🔎 Casos probables", min_value=0, value=150)
        hospitalized_count = st.number_input("🏥 Hospitalizados", min_value=0, value=50)
        case_count_7day_avg = st.number_input(
            "📊 Promedio 7 días (casos)", min_value=0.0, value=180.0
        )

    with col2:
        all_case_count_7day_avg = st.number_input(
            "📊 Promedio 7 días (todos los casos)", min_value=0.0, value=300.0
        )
        hosp_count_7day_avg = st.number_input(
            "📈 Promedio 7 días (hospitalizados)", min_value=0.0, value=40.0
        )
        death_count_7day_avg = st.number_input(
            "💀 Promedio 7 días (muertes)", min_value=0.0, value=10.0
        )

    data = {
        "case_count": case_count,
        "probable_case_count": probable_case_count,
        "hospitalized_count": hospitalized_count,
        "case_count_7day_avg": case_count_7day_avg,
        "all_case_count_7day_avg": all_case_count_7day_avg,
        "hosp_count_7day_avg": hosp_count_7day_avg,
        "death_count_7day_avg": death_count_7day_avg,
    }

    return pd.DataFrame(data, index=[0])


def main() -> None:
    st.title("📉 Predicción de muertes por COVID-19 en NYC")
    st.markdown(
        "Este modelo predice la cantidad de muertes en función de "
        + "variables epidemiológicas oficiales de la ciudad de Nueva York."
    )

    model = load_model()
    input_df = user_input_form()

    prediction = model.predict(input_df)[0]
    prediction = round(prediction)

    st.subheader("🔮 Resultado de la predicción")
    st.metric("Predicción de muertes", f"{prediction} fallecimientos")

    if prediction > HIGH_DEATH_THRESHOLD:
        st.error("⚠️ Nivel alto de mortalidad esperada")
    elif prediction > MEDIUM_DEATH_THRESHOLD:
        st.warning("⚠️ Nivel medio de mortalidad esperada")
    else:
        st.success("✅ Nivel bajo de mortalidad esperada")


if __name__ == "__main__":
    main()
