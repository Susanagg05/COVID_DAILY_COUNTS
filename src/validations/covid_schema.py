import pandera as pa
from pandera import Check, Column, DataFrameSchema

covid_schema = DataFrameSchema(
    {
        "date_of_interest": Column(
            pa.DateTime,
            nullable=False,
            coerce=True,
            description="Fecha de referencia para los conteos diarios",
        ),
        "CASE_COUNT": Column(
            pa.Int64,
            checks=Check.ge(0),
            nullable=False,
            description="Número total de casos confirmados",
        ),
        "PROBABLE_CASE_COUNT": Column(
            pa.Int64, checks=Check.ge(0), nullable=False, description="Número de casos probables"
        ),
        "HOSPITALIZED_COUNT": Column(
            pa.Int64, checks=Check.ge(0), nullable=False, description="Número de hospitalizaciones"
        ),
        "DEATH_COUNT": Column(
            pa.Int64, checks=Check.ge(0), nullable=False, description="Número de muertes"
        ),
        "CASE_COUNT_7DAY_AVG": Column(
            pa.Int64,
            checks=Check.ge(0),
            nullable=False,
            description="Promedio móvil de 7 días de casos confirmados",
        ),
        "ALL_CASE_COUNT_7DAY_AVG": Column(
            pa.Int64,
            checks=Check.ge(0),
            nullable=False,
            description="Promedio móvil de 7 días de todos los casos (confirmados + probables)",
        ),
        "HOSP_COUNT_7DAY_AVG": Column(
            pa.Int64,
            checks=Check.ge(0),
            nullable=False,
            description="Promedio móvil de 7 días de hospitalizaciones",
        ),
        "DEATH_COUNT_7DAY_AVG": Column(
            pa.Int64,
            checks=Check.ge(0),
            nullable=False,
            description="Promedio móvil de 7 días de muertes",
        ),
    },
    strict=True,
    coerce=True,
    name="CovidDailyCountsSchema",
)
