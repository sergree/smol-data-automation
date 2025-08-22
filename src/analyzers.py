"""
Модуль для анализа данных и обучения моделей машинного обучения.

Содержит классы для вычисления статистик и тренировки ML моделей.
"""

from abc import ABC, abstractmethod

import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose

from src.config.pipelines import Pipeline, Stats, TimeSeriesStats


class BaseAnalyzer(ABC):
    """Базовый абстрактный класс для анализаторов данных."""

    @abstractmethod
    def analyze(self, df: pd.DataFrame, pipeline: Pipeline):
        """Анализирует данные согласно пайплайну."""
        pass


class MakeDefaultStats(BaseAnalyzer):
    """Вычисляет базовые статистики для числовых колонок."""

    def analyze(self, df: pd.DataFrame, pipeline: Pipeline):
        """Вычисляет среднее, медиану, моду и стандартное отклонение."""
        mean = df.mean()
        logger.info(f"Среднее:\n{mean.to_string()}")

        median = df.median()
        logger.info(f"Медиана:\n{median.to_string()}")

        mode = df.mode().iloc[0]
        logger.info(f"Мода:\n{mode.to_string()}")

        std = df.std()
        logger.info(f"Стандартное отклонение:\n{std.to_string()}")

        pipeline.meta.stats = Stats(mean=mean, median=median, mode=mode, std=std)
        logger.info("Значения статистик сохранены")


class MakeTimeSeriesStats(BaseAnalyzer):
    """Анализирует временные ряды и извлекает компоненты."""

    def analyze(self, df: pd.DataFrame, pipeline: Pipeline):
        """Выполняет декомпозицию временного ряда на тренд и сезонность."""
        if not pipeline.meta.is_timeseries:
            logger.info("Пропускаю анализ временных рядов для невременного ряда")
            return

        target_col = pipeline.model.target

        if pipeline.time_period:
            decomp = seasonal_decompose(df[target_col], period=pipeline.time_period)
        else:
            decomp = seasonal_decompose(df[target_col])

        trend = decomp.trend
        seasonal = decomp.seasonal

        pipeline.meta.timeseries_stats = TimeSeriesStats(trend=trend, seasonal=seasonal)
        logger.info("Значения кривых тренда и сезонности сохранены")


class TrainModel(BaseAnalyzer):
    """Обучает модель машинного обучения на подготовленных данных."""

    def analyze(self, df: pd.DataFrame, pipeline: Pipeline):
        """Обучает модель и вычисляет метрики производительности."""
        X = df.drop(columns=[pipeline.model.target])
        y = df[pipeline.model.target]
        logger.info(
            f"Данные разделены на признаки и целевую переменную {pipeline.model.target}"
        )

        if pipeline.model.random_state is not None:
            logger.info(
                f"Для воспроизводимости установлен random_state = {pipeline.model.random_state}"
            )
        else:
            logger.info(
                "random_state не установлен, для воспроизводимости вы можете настроить random_state в конфиге"
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=pipeline.model.test_size,
            random_state=pipeline.model.random_state,
        )
        logger.info(
            f"Обучающая выборка: {len(X_train)} образцов, тестовая: {len(X_test)} образцов (соотношение {pipeline.model.test_size})"
        )

        if pipeline.model.type == "regression":
            model = RandomForestRegressor(random_state=pipeline.model.random_state)
        elif pipeline.model.type == "classification":
            model = RandomForestClassifier(random_state=pipeline.model.random_state)
        elif pipeline.model.type == "forecasting":
            model = RandomForestRegressor(random_state=pipeline.model.random_state)

        logger.info(
            f"Для режима {pipeline.model.type} используется модель {type(model).__name__}"
        )

        logger.info(f"Запускаю обучение модели {type(model).__name__}...")
        model.fit(X_train, y_train)
        logger.info(f"Модель {type(model).__name__} успешно обучена")
        y_pred = model.predict(X_test)

        if pipeline.model.type in ["regression", "forecasting"]:
            metrics = {
                "rmse": root_mean_squared_error(y_test, y_pred),
                "mae": mean_absolute_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred),
            }
        elif pipeline.model.type == "classification":
            y_pred_proba = (
                model.predict_proba(X_test)[:, 1] if len(model.classes_) == 2 else None
            )
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
                "recall": recall_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
                "f1": f1_score(y_test, y_pred, average="weighted"),
            }
            if y_pred_proba is not None:
                metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba)
        logger.info(f"Метрики для режима {pipeline.model.type}: {metrics}")

        pipeline.meta.trained_model = model
        pipeline.meta.metrics = metrics

        logger.info("Обученная модель и значения метрик сохранены")
