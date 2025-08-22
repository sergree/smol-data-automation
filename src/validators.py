"""
Модуль для валидации и очистки данных.

Содержит классы для проверки качества данных и их исправления.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from src.config.pipelines import Pipeline


class BaseValidator(ABC):
    """Базовый абстрактный класс для валидаторов данных."""

    def __init__(self):
        """Инициализирует валидатор с пустым результатом проверки."""
        self.validation_result = None

    @abstractmethod
    def validate(self, df: pd.DataFrame, pipeline: Pipeline):
        """Проверяет данные и сохраняет результат."""
        pass

    @abstractmethod
    def fix(self, df: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
        """Исправляет найденные проблемы в данных."""
        pass


class FullDuplicates(BaseValidator):
    """Проверяет и удаляет полные дубликаты строк."""

    def validate(self, df: pd.DataFrame, pipeline: Pipeline):
        """Подсчитывает количество полных дубликатов."""
        if pipeline.meta.is_timeseries:
            logger.info("Пропускаю проверку дубликатов для временного ряда")
            return self
        duplicates_count = df.duplicated().sum()
        logger.info(f"Найдено полных дубликатов: {duplicates_count} из {len(df)} строк")
        self.validation_result = duplicates_count
        return self

    def fix(self, df: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
        """Удаляет найденные дубликаты."""
        if pipeline.meta.is_timeseries:
            return df
        if self.validation_result > 0:
            df = df.drop_duplicates()
            logger.info(
                f"Удалено {self.validation_result} полных дубликатов, осталось {len(df)} уникальных строк"
            )
        return df


class WrongTypes(BaseValidator):
    """Проверяет и исправляет неправильные типы данных."""

    def validate(self, df: pd.DataFrame, pipeline: Pipeline):
        """Находит столбцы, которые можно конвертировать в числовые."""
        convertible_cols = []
        threshold = 0.95

        for col in df.select_dtypes(exclude=[np.number]).columns:
            test_convert = pd.to_numeric(df[col], errors="coerce")
            success_rate = test_convert.notna().sum() / len(df)
            if success_rate >= threshold:
                convertible_cols.append(col)

        logger.info(
            f"Столбцы с неверным типом для преобразования в числовые: {len(convertible_cols) if len(convertible_cols) else 'НЕ НАЙДЕНЫ'}"
        )
        self.validation_result = convertible_cols
        return self

    def fix(self, df: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
        """Конвертирует найденные столбцы в числовые."""
        for col in self.validation_result:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            logger.info(f"Столбец {col} преобразован в числовой")
        return df


class NaNs(BaseValidator):
    """Проверяет и обрабатывает пропущенные значения."""

    def validate(self, df: pd.DataFrame, pipeline: Pipeline):
        """Подсчитывает пропуски в каждом столбце."""
        missing_info = df.isnull().sum()
        missing_cols = missing_info[missing_info > 0].index.tolist()
        logger.info(
            f"Столбцы с пропусками: {missing_cols if missing_cols else 'НЕ НАЙДЕНЫ'}"
        )
        self.validation_result = missing_info
        return self

    def fix(self, df: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
        """Обрабатывает пропуски согласно стратегии пайплайна."""
        if self.validation_result.sum() == 0:
            return df

        strategy = pipeline.transform.missing_strategy

        if strategy == "drop":
            df = df.dropna()
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if strategy == "mean":
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif strategy == "median":
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

            df = df.dropna()

        logger.info(f"Пропуски обработаны стратегией {strategy}")
        return df


class Outliers(BaseValidator):
    """Проверяет и удаляет выбросы в данных."""

    def _get_outlier_mask(self, df: pd.DataFrame, strategy: str) -> pd.Series:
        """Создает маску для выбросов согласно выбранной стратегии."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return pd.Series([False] * len(df), index=df.index)

        if strategy == "iqr":
            quantiles = df[numeric_cols].quantile([0.25, 0.75])
            Q1, Q3 = quantiles.iloc[0], quantiles.iloc[1]
            IQR = Q3 - Q1
            outliers_df = (df[numeric_cols] < (Q1 - 1.5 * IQR)) | (
                df[numeric_cols] > (Q3 + 1.5 * IQR)
            )
        else:
            z_scores = np.abs(stats.zscore(df[numeric_cols], nan_policy="omit"))
            outliers_array = z_scores > 3
            outliers_df = pd.DataFrame(
                outliers_array, index=df.index, columns=numeric_cols
            )

        return outliers_df.any(axis=1).fillna(False)

    def validate(self, df: pd.DataFrame, pipeline: Pipeline):
        """Находит выбросы согласно выбранной стратегии."""
        if pipeline.meta.is_timeseries:
            logger.info("Пропускаю проверку выбросов для временного ряда")
            return self
        strategy = pipeline.transform.outlier_strategy
        self.outlier_mask = self._get_outlier_mask(df, strategy)
        self.validation_result = self.outlier_mask.sum()

        logger.info(f"Найдено {self.validation_result} выбросов методом {strategy}")
        return self

    def fix(self, df: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
        """Удаляет найденные выбросы."""
        if pipeline.meta.is_timeseries:
            return df
        if self.validation_result == 0:
            return df

        df_clean = df[~self.outlier_mask]
        strategy = pipeline.transform.outlier_strategy

        logger.info(
            f"Выбросы удалены методом {strategy}, осталось {len(df_clean)} строк"
        )
        return df_clean
