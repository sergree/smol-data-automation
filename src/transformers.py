"""
Модуль для трансформации данных.

Содержит классы для предобработки и преобразования данных перед анализом.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.config.pipelines import Pipeline


class BaseTransformer(ABC):
    """Базовый абстрактный класс для трансформаторов данных."""

    @abstractmethod
    def transform(self, df: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
        """Трансформирует DataFrame согласно настройкам пайплайна."""
        pass


class DropColumnsManual(BaseTransformer):
    """Удаляет или оставляет указанные столбцы."""

    def transform(self, df: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
        """Удаляет ненужные столбцы или оставляет только выбранные."""
        if pipeline.transform.pick_columns:
            logger.info(
                f"Сразу же оставляем только столбцы: {pipeline.transform.pick_columns}"
            )
            columns = [
                col
                for col in pipeline.transform.pick_columns
                if col not in pipeline.transform.drop_columns
            ]
            df = df[columns]
        elif pipeline.transform.drop_columns:
            logger.info(f"Сразу же удаляем столбцы: {pipeline.transform.drop_columns}")
            df = df.drop(columns=pipeline.transform.drop_columns)
        return df


class SetTimeColumn(BaseTransformer):
    """Устанавливает временной столбец и конвертирует в временной ряд."""

    def transform(self, df: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
        """Конвертирует столбец времени и устанавливает его как индекс."""
        if pipeline.time_column:
            logger.info(
                f"Обнаружен столбец, который будет использоваться в качестве времени: {pipeline.time_column}"
            )
            df[pipeline.time_column] = pd.to_datetime(
                df[pipeline.time_column], utc=True
            )
            logger.info(
                f"{pipeline.time_column} автоматически сконвертирован в datetime"
            )
            df = df.set_index(pipeline.time_column).sort_index()
            logger.info(f"{pipeline.time_column} установлен в качестве индекса")
            pipeline.meta.is_timeseries = True
        return df


class Encode(BaseTransformer):
    """Кодирует категориальные признаки в числовые."""

    def transform(self, df: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
        """Применяет Label и OneHot кодирование к указанным столбцам."""
        label_cols = pipeline.transform.encode.label
        onehot_cols = pipeline.transform.encode.onehot
        if not label_cols and not onehot_cols:
            logger.info("Пропускаю кодирование признаков, поля не заданы пользователем")
            return df

        for col in label_cols:
            logger.info(f"Кодирую колонку {col} с помощью LabelEncoder")
            df[col] = LabelEncoder().fit_transform(df[col])
            logger.info(f"Новые уникальные значения {col}: {df[col].unique()}")

        for col in onehot_cols:
            logger.info(f"Кодирую колонку {col} с помощью OneHotEncoder")
            encoder = OneHotEncoder(sparse_output=False)
            encoded = encoder.fit_transform(df[[col]])
            feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]

            df_encoded = pd.DataFrame(encoded, columns=feature_names, index=df.index)
            df.drop(col, axis=1, inplace=True)
            for i, fname in enumerate(feature_names):
                df[fname] = df_encoded.iloc[:, i]
            logger.info(f"Новые столбцы вместо {col}: {feature_names}")

        return df


class DropNotNumeric(BaseTransformer):
    """Удаляет нечисловые столбцы, которые не были обработаны."""

    def transform(self, df: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
        """Удаляет нечисловые столбцы с предупреждением."""
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

        for col in non_numeric_cols:
            unique_count = df[col].nunique()
            sample_values = df[col].unique()[:3]
            logger.warning(
                f"Удалена необработанная нечисловая колонка: {col} ({unique_count} уник. значений, примеры: {sample_values})"
            )

        if len(non_numeric_cols) > 0:
            df = df.select_dtypes(include=[np.number])
            logger.warning(
                "Добавьте их в раздел transform файла конфигурации пайплайна"
            )
            logger.warning(
                "Либо в encode.label, либо в encode.one_hot, либо в drop_columns"
            )

        return df


class Scale(BaseTransformer):
    """Масштабирует числовые признаки с помощью StandardScaler."""

    def transform(self, df: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
        """Применяет стандартизацию к числовым столбцам."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = pipeline.transform.no_scale

        target_col = pipeline.model.target
        if target_col in numeric_cols:
            exclude_cols = exclude_cols + [target_col]
            logger.info(
                f"Целевая переменная `{target_col}` исключена из масштабирования"
            )

        cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]

        if len(cols_to_scale) == 0:
            logger.info("Нет числовых колонок для масштабирования")
            return df

        if exclude_cols:
            logger.info(f"Исключены из масштабирования: {exclude_cols}")

        logger.info(f"Масштабирую числовые колонки: {cols_to_scale}")
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

        return df


class PrepareForecastingFeatures(BaseTransformer):
    """Подготавливает признаки для прогнозирования временных рядов."""

    def transform(self, df: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
        """Добавляет лаги и скользящие средние для временных рядов."""
        if not pipeline.meta.is_timeseries or pipeline.model.type != "forecasting":
            return df

        logger.info("Подготавливаю временные признаки для режима forecasting")
        target_col = pipeline.model.target

        df[f"{target_col}_trend"] = pipeline.meta.timeseries_stats.trend
        df[f"{target_col}_seasonal"] = pipeline.meta.timeseries_stats.seasonal

        df[f"{target_col}_lag_1"] = df[target_col].shift(1)
        df[f"{target_col}_lag_2"] = df[target_col].shift(2)
        df[f"{target_col}_lag_3"] = df[target_col].shift(3)

        df[f"{target_col}_ma_3"] = df[target_col].rolling(window=3).mean()

        df = df.dropna()
        logger.info("Пропуски временного смещения обработаны стратегией drop")

        return df
