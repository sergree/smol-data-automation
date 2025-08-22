"""
Модуль конфигурации пайплайнов.

Содержит Pydantic модели для описания пайплайнов обработки данных.
"""

from typing import Annotated, Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class CSVFileDataSource(BaseModel):
    """Источник данных из CSV файла."""

    type: Literal["csv_file"]
    path: str


class CSVUrlDataSource(BaseModel):
    """Источник данных из CSV по URL."""

    type: Literal["csv_url"]
    url: str


class ExcelFileDataSource(BaseModel):
    """Источник данных из Excel файла."""

    type: Literal["excel_file"]
    path: str


class PostgreSQLDataSource(BaseModel):
    """Источник данных из PostgreSQL базы."""

    type: Literal["postgresql"]
    connection_name: str
    sql_query: str


class RestAPIDataSource(BaseModel):
    """Источник данных из REST API."""

    type: Literal["rest_api"]
    url: str


class JSONFileDataSource(BaseModel):
    """Источник данных из JSON файла."""

    type: Literal["json_file"]
    path: str


class Encode(BaseModel):
    """Настройки кодирования категориальных признаков."""

    label: list[str] = []
    onehot: list[str] = []


class Transform(BaseModel):
    """Настройки трансформации данных."""

    drop_columns: list[str] = []
    pick_columns: list[str] = []
    missing_strategy: Literal["drop", "mean", "median"] = "drop"
    outlier_strategy: Literal["iqr", "zscore"] = "iqr"
    no_scale: list[str] = []
    encode: Encode = Field(default_factory=Encode)


class Model(BaseModel):
    """Настройки модели машинного обучения."""

    type: Literal["regression", "classification", "forecasting"]
    target: str
    test_size: float = 0.2
    random_state: int | None = None


class Stats(BaseModel):
    """Статистики числовых признаков."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    mean: pd.Series
    median: pd.Series
    mode: pd.Series
    std: pd.Series


class TimeSeriesStats(BaseModel):
    """Статистики временного ряда."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    trend: pd.Series
    seasonal: pd.Series


class ComputedMeta(BaseModel):
    """Вычисляемые метаданные пайплайна."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    is_timeseries: bool = False
    stats: Stats | None = None
    timeseries_stats: TimeSeriesStats | None = None
    trained_model: Any | None = None
    metrics: dict[str, float] | None = None
    run_id: str | None = None
    results_path: str | None = None


class Report(BaseModel):
    """Настройки отчетности."""

    smtp_connection_name: str | None = None
    email_recipients: list[str] | None = None
    postgresql_connection_name: str | None = None
    bi_custom_api_connection_name: str | None = None


class Pipeline(BaseModel):
    """Полная конфигурация пайплайна обработки данных."""

    name: str
    meta: ComputedMeta = Field(default_factory=ComputedMeta)
    data_source: Annotated[
        CSVFileDataSource
        | CSVUrlDataSource
        | ExcelFileDataSource
        | PostgreSQLDataSource
        | RestAPIDataSource
        | JSONFileDataSource,
        Field(discriminator="type"),
    ]
    time_column: str | None = None
    time_period: int | None = None
    transform: Transform = Field(default_factory=Transform)
    model: Model
    report: Report = Field(default_factory=Report)


class PipelinesConfig(BaseModel):
    """Конфигурация всех пайплайнов приложения."""

    pipelines: list[Pipeline]
