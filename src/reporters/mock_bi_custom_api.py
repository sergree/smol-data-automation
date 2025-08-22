"""
Мок-модуль для интеграции с системами бизнес-аналитики.

Демонстрирует структуру данных для отправки в BI системы через REST API.
"""

import json
from datetime import datetime
from typing import Any, Dict

import pandas as pd
from loguru import logger

from src.config import AppConfig
from src.config.pipelines import Pipeline
from src.reporters.base import BaseReporter
from src.reporters.utils import serialize_stats, serialize_timeseries_stats


class MockBICustomAPIReporter(BaseReporter):
    """
    Мок-репортер для интеграции с системами бизнес-аналитики через REST API.

    Этот класс демонстрирует как подготавливать и структурировать данные для BI систем.
    Для реальной интеграции замените мок-методы на настоящие HTTP запросы
    к REST API endpoints вашей BI системы.
    """

    def __init__(self, *args, **kwargs):
        """Инициализирует репортер с пустыми API URL и токеном."""
        super().__init__(*args, **kwargs)
        self.api_url = None
        self.token = None

    async def report(
        self,
        pipeline: Pipeline,
        df: pd.DataFrame,
        df_no_scale: pd.DataFrame,
        config: AppConfig,
    ):
        """Отправляет данные в BI систему через мок API вызовы."""
        if not pipeline.report.bi_custom_api_connection_name:
            logger.info(
                "[МОК] Результаты не будут сохранены в систему бизнес-анализа т.к. не указано API-соединение"
            )
            return
        connection = config.get_bi_custom_api_connection(
            pipeline.report.bi_custom_api_connection_name
        )
        self.api_url = connection.api_url
        self.token = connection.token
        payload = self._prepare_bi_payload(pipeline, df, df_no_scale)

        # Мок API вызовы - замените на реальные HTTP запросы
        await self._send_model_metrics(payload["model_metrics"])
        await self._send_dataset_stats(payload["dataset_stats"])
        await self._send_predictions_sample(payload["predictions_sample"])

        logger.info(
            f"Интеграция с BI системой завершена для пайплайна: {pipeline.name}"
        )

    def _prepare_bi_payload(
        self, pipeline: Pipeline, df: pd.DataFrame, df_no_scale: pd.DataFrame
    ) -> Dict[str, Any]:
        """Подготавливает структурированные данные для API BI системы."""

        # Базовая информация о пайплайне
        base_info = {
            "pipeline_name": pipeline.name,
            "run_id": pipeline.meta.run_id,
            "model_type": pipeline.model.type,
            "target_column": pipeline.model.target,
            "timestamp": datetime.now().isoformat(),
            "is_timeseries": pipeline.meta.is_timeseries,
        }

        # Метрики производительности модели
        model_metrics = {
            **base_info,
            "metrics": pipeline.meta.metrics or {},
            "test_size": pipeline.model.test_size,
            "model_class": type(pipeline.meta.trained_model).__name__
            if pipeline.meta.trained_model
            else None,
        }

        # Статистики датасета
        dataset_stats = {
            **base_info,
            "dataset_shape": df_no_scale.shape,
            "features_count": len(df_no_scale.columns) - 1,
            "missing_strategy": pipeline.transform.missing_strategy,
            "outlier_strategy": pipeline.transform.outlier_strategy,
        }

        if pipeline.meta.stats:
            dataset_stats["stats"] = serialize_stats(pipeline.meta.stats)

        if pipeline.meta.is_timeseries and pipeline.meta.timeseries_stats:
            dataset_stats["timeseries_stats"] = serialize_timeseries_stats(
                pipeline.meta.timeseries_stats
            )

        # Образцы предсказаний для визуализации
        predictions_sample = {
            **base_info,
            "sample_data": df_no_scale.head(10).to_dict("records"),
            "columns_info": {
                "total_columns": len(df_no_scale.columns),
                "encoded_columns": pipeline.transform.encode.label
                + pipeline.transform.encode.onehot,
                "dropped_columns": pipeline.transform.drop_columns,
            },
        }

        return {
            "model_metrics": model_metrics,
            "dataset_stats": dataset_stats,
            "predictions_sample": predictions_sample,
        }

    async def _send_model_metrics(self, data: Dict[str, Any]):
        """Мок: Отправка метрик модели в BI дашборд."""
        endpoint = f"{self.api_url}/models/metrics"

        # TODO: Заменить на реальный HTTP запрос
        # response = await httpx.post(endpoint, json=data, headers=self._get_headers())

        logger.info(f"[МОК] Отправка метрик модели в {endpoint}")
        logger.debug(f"Данные: {json.dumps(data, indent=2, ensure_ascii=False)}")

    async def _send_dataset_stats(self, data: Dict[str, Any]):
        """Мок: Отправка статистик датасета в BI аналитику."""
        endpoint = f"{self.api_url}/datasets/stats"

        # TODO: Заменить на реальный HTTP запрос
        # response = await httpx.post(endpoint, json=data, headers=self._get_headers())

        logger.info(f"[МОК] Отправка статистик датасета в {endpoint}")
        logger.debug(
            f"Размер датасета: {data.get('dataset_shape')}, Признаков: {data.get('features_count')}"
        )

    async def _send_predictions_sample(self, data: Dict[str, Any]):
        """Мок: Отправка образцов предсказаний для BI визуализации."""
        endpoint = f"{self.api_url}/predictions/sample"

        # TODO: Заменить на реальный HTTP запрос
        # response = await httpx.post(endpoint, json=data, headers=self._get_headers())

        logger.info(f"[МОК] Отправка образцов предсказаний в {endpoint}")
        logger.debug(f"Размер выборки: {len(data.get('sample_data', []))}")

    def _get_headers(self) -> Dict[str, str]:
        """Получает HTTP заголовки для запросов к BI API."""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers
