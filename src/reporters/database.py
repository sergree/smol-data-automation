"""
Модуль для сохранения результатов в PostgreSQL базу данных.

Содержит классы для работы с базой данных и сохранения результатов ML пайплайнов.
"""

from datetime import datetime

import pandas as pd
from loguru import logger
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from src.config import AppConfig
from src.config.pipelines import Pipeline
from src.reporters.base import BaseReporter
from src.reporters.utils import (
    serialize_model,
    serialize_stats,
    serialize_timeseries_stats,
)

Base = declarative_base()


class Run(Base):
    """Таблица для хранения информации о запусках пайплайнов."""

    __tablename__ = "ml_runs"

    id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class PipelineResult(Base):
    """Таблица для хранения результатов выполнения пайплайнов."""

    __tablename__ = "ml_pipeline_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, nullable=False)
    pipeline_name = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    target_column = Column(String, nullable=False)
    test_size = Column(Float, nullable=False)
    is_timeseries = Column(Boolean, default=False)
    metrics = Column(JSON)
    stats = Column(JSON)
    timeseries_stats = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class PipelineResultExt(Base):
    """Расширенная таблица для хранения сериализованных моделей."""

    __tablename__ = "ml_pipeline_results_ext"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pipeline_result_id = Column(
        Integer, ForeignKey("ml_pipeline_results.id"), nullable=False
    )
    model_data = Column(LargeBinary)


class PostgreSQLReporter(BaseReporter):
    """Репортер для сохранения результатов в PostgreSQL базу данных."""

    def __init__(self, *args, **kwargs):
        """Инициализирует репортер с пустыми engine и session_factory."""
        super().__init__(*args, **kwargs)
        self.engine = None
        self.session_factory = None

    def _convert_to_async_url(self, url: str) -> str:
        """Конвертирует обычный PostgreSQL URL в асинхронный."""
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://")
        return url

    async def _init_engine(self, connection_string: str):
        """Инициализирует асинхронный engine и создает таблицы."""
        async_url = self._convert_to_async_url(connection_string)
        self.engine = create_async_engine(async_url, echo=False)
        self.session_factory = sessionmaker(self.engine, class_=AsyncSession)

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def report(
        self,
        pipeline: Pipeline,
        df: pd.DataFrame,
        df_no_scale: pd.DataFrame,
        config: AppConfig,
    ):
        """Сохраняет результаты пайплайна в базу данных."""
        if not pipeline.report.postgresql_connection_name:
            logger.info(
                "Результаты не будут сохранены в базу данных т.к. не указано PostgreSQL-соединение"
            )
            return

        logger.info("Сохраняю результаты анализа и моделей в базу данных...")
        conn_config = config.get_postresql_connection(
            pipeline.report.postgresql_connection_name
        )
        await self._init_engine(conn_config.connection_string)

        async with self.session_factory() as session:
            try:
                run_record = Run(id=pipeline.meta.run_id)
                session.add(run_record)
                await session.flush()
            except IntegrityError:
                await session.rollback()

            pipeline_result = PipelineResult(
                run_id=pipeline.meta.run_id,
                pipeline_name=pipeline.name,
                model_type=pipeline.model.type,
                target_column=pipeline.model.target,
                test_size=pipeline.model.test_size,
                is_timeseries=pipeline.meta.is_timeseries,
                metrics=pipeline.meta.metrics,
                stats=serialize_stats(pipeline.meta.stats),
                timeseries_stats=serialize_timeseries_stats(
                    pipeline.meta.timeseries_stats
                ),
            )

            session.add(pipeline_result)
            await session.flush()

            if pipeline.meta.trained_model is not None:
                model_data = serialize_model(pipeline.meta.trained_model)
                pipeline_ext = PipelineResultExt(
                    pipeline_result_id=pipeline_result.id, model_data=model_data
                )
                session.add(pipeline_ext)

            await session.commit()

        await self.engine.dispose()
        logger.info(
            "Результаты анализа и моделей успешно сохранены в таблицы ml_runs, ml_pipeline_results, ml_pipeline_results_ext базы данных"
        )
