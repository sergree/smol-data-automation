"""
Точка входа в приложение автоматизации обработки данных.

Запускает пайплайны обработки данных на основе конфигурации.
"""

import argparse
import asyncio
import os
from pathlib import Path

from loguru import logger

from src.analyzers import MakeDefaultStats, MakeTimeSeriesStats, TrainModel
from src.config import AppConfig
from src.config.pipelines import Pipeline
from src.extractors import get_extractor
from src.reporters import (
    CompositeFileEmailReporter,
    MockBICustomAPIReporter,
    PostgreSQLReporter,
)
from src.transformers import (
    DropColumnsManual,
    DropNotNumeric,
    Encode,
    PrepareForecastingFeatures,
    Scale,
    SetTimeColumn,
)
from src.validators import FullDuplicates, NaNs, Outliers, WrongTypes

logger.add("logs/run_{time}.log", backtrace=True, diagnose=True)


class App:
    """Основной класс приложения для запуска пайплайнов обработки данных."""

    @logger.catch(reraise=True)
    def __init__(self, config: AppConfig):
        """Инициализация приложения с конфигурацией."""
        logger.info("Читаю данные из конфигов...")
        self.config = config
        logger.info("Данные из конфигов загружены")
        self.pipeline_names_all = set(x.name for x in self.config.pipelines)
        self.pipeline_names_ok = set()

    def set_ok(self, pipeline: Pipeline):
        """Отмечает пайплайн как успешно выполненный."""
        self.pipeline_names_ok.add(pipeline.name)

    @logger.catch(reraise=True)
    async def run(self):
        """Запускает все пайплайны из конфигурации."""
        logger.info("Скрипт автоматизации обработки данных запущен")
        logger.info(f"ID запуска {self.get_run_id()}")
        logger.info(f"Количество пайплайнов: {len(self.pipeline_names_all)}")
        for pipeline in self.config.pipelines:
            await self.run_one(pipeline)
        pipeline_names_errored = self.pipeline_names_all - self.pipeline_names_ok
        logger.info(
            f"Итог выполнения cкрипта автоматизации обработки данных: {len(self.pipeline_names_ok)}/{len(self.pipeline_names_all)} пайплайнов успешно"
        )
        if not pipeline_names_errored:
            logger.info("Выполнено успешно")
        else:
            logger.warning(
                f"Завершено С ОШИБКАМИ в пайплайнах: {', '.join(pipeline_names_errored)}"
            )

    @staticmethod
    def get_run_id() -> str:
        """Получает ID текущего запуска из имени лог-файла."""
        for handler in logger._core.handlers.values():
            if hasattr(handler._sink, "_file") and handler._sink._file:
                return Path(handler._sink._file.name).stem

    @staticmethod
    def get_results_path(pipeline_name: str) -> str:
        """Формирует путь для сохранения результатов пайплайна."""
        return Path("results") / App.get_run_id() / pipeline_name

    @logger.catch
    async def run_one(self, pipeline: Pipeline):
        """Выполняет один пайплайн: извлечение, трансформация, анализ и загрузка."""
        logger.info(f"Запускаю пайплайн `{pipeline.name}`...")

        # 1. Extract

        logger.info(
            f"Инициализирую экстрактор для {type(pipeline.data_source).__name__}..."
        )
        extractor = get_extractor(pipeline.data_source, self.config)
        logger.info(f"Экстрактор {type(extractor).__name__} инициализирован")

        logger.info("Получаю данные в формате pandas.DataFrame...")
        df = await extractor.extract()
        logger.info("Данные успешно получены")

        # 2. Transform

        df = DropColumnsManual().transform(df, pipeline)
        df = SetTimeColumn().transform(df, pipeline)
        logger.info(f"Количество строк до очистки датасета: {len(df)}")
        logger.debug(f"pipeline.meta.is_timeseries = {pipeline.meta.is_timeseries}")

        logger.debug(f"Первые несколько значений:\n{df.head().to_string()}")

        df = FullDuplicates().validate(df, pipeline).fix(df, pipeline)
        df = WrongTypes().validate(df, pipeline).fix(df, pipeline)
        df = NaNs().validate(df, pipeline).fix(df, pipeline)
        df = Outliers().validate(df, pipeline).fix(df, pipeline)

        df = Encode().transform(df, pipeline)
        df = DropNotNumeric().transform(df, pipeline)

        logger.debug(
            f"Первые несколько значений после исправления:\n{df.head().to_string()}"
        )

        MakeDefaultStats().analyze(df, pipeline)
        MakeTimeSeriesStats().analyze(df, pipeline)

        df_no_scale = df.copy()

        df = Scale().transform(df, pipeline)
        df = PrepareForecastingFeatures().transform(df, pipeline)

        logger.info(f"Количество строк после очистки датасета: {len(df)}")
        logger.debug(
            f"Первые несколько значений после масштабирования:\n{df.head().to_string()}"
        )

        TrainModel().analyze(df, pipeline)

        # 3. Load

        pipeline.meta.run_id = self.get_run_id()
        pipeline.meta.results_path = self.get_results_path(pipeline.name)
        logger.info(f"Результаты будут сохранены в папку {pipeline.meta.results_path}")
        os.makedirs(pipeline.meta.results_path, exist_ok=True)

        await CompositeFileEmailReporter().report(
            pipeline, df, df_no_scale, self.config
        )
        await PostgreSQLReporter().report(pipeline, df, df_no_scale, self.config)
        await MockBICustomAPIReporter().report(pipeline, df, df_no_scale, self.config)

        self.set_ok(pipeline)
        logger.info(f"Пайплайн `{pipeline.name}` завершён успешно")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Автоматизация обработки данных")
    parser.add_argument(
        "--connections-yaml",
        default="./config/connections.yaml",
        help="Путь до YAML-конфига соединений (по умолчанию: ./config/connections.yaml)",
    )
    parser.add_argument(
        "--pipelines-yaml",
        default="./config/pipelines.yaml",
        help="Путь до YAML-конфига пайплайнов (по умолчанию: ./config/pipelines.yaml)",
    )

    args = parser.parse_args()

    asyncio.run(
        App(
            AppConfig(
                connections_yaml_path=args.connections_yaml,
                pipelines_yaml_path=args.pipelines_yaml,
            )
        ).run()
    )
