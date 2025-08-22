"""
Базовые классы для репортеров.

Содержит абстрактные классы для создания отчетов.
"""

from abc import ABC, abstractmethod

import pandas as pd

from src.config import AppConfig
from src.config.pipelines import Pipeline


class BaseReporter(ABC):
    """Базовый абстрактный класс для всех репортеров."""

    @abstractmethod
    async def report(
        self,
        pipeline: Pipeline,
        df: pd.DataFrame,
        df_no_scale: pd.DataFrame,
        config: AppConfig,
    ):
        """Создает отчет на основе результатов пайплайна."""
        pass
