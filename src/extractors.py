"""
Модуль для извлечения данных из различных источников.

Содержит классы для загрузки данных из файлов, баз данных и API.
"""

import json
from abc import ABC, abstractmethod
from io import StringIO
from typing import Any

import asyncpg
import httpx
import pandas as pd

from src.config import AppConfig


class BaseExtractor(ABC):
    """Базовый абстрактный класс для извлечения данных."""

    def __init__(self, data_source: Any, config: AppConfig):
        """Инициализирует экстрактор с источником данных и конфигурацией."""
        self.data_source = data_source
        self.config = config

    @abstractmethod
    async def extract(self) -> pd.DataFrame:
        """Извлекает данные и возвращает DataFrame."""
        pass


class CSVFile(BaseExtractor):
    """Экстрактор для CSV файлов."""

    async def extract(self) -> pd.DataFrame:
        """Загружает данные из CSV файла."""
        return pd.read_csv(self.data_source.path)


class CSVUrl(BaseExtractor):
    """Экстрактор для CSV по URL."""

    async def extract(self) -> pd.DataFrame:
        """Загружает CSV данные по HTTP URL."""
        async with httpx.AsyncClient() as client:
            response = await client.get(self.data_source.url)
            return pd.read_csv(StringIO(response.text))


class ExcelFile(BaseExtractor):
    """Экстрактор для Excel файлов."""

    async def extract(self) -> pd.DataFrame:
        """Загружает данные из Excel файла."""
        return pd.read_excel(self.data_source.path)


class PostgreSQL(BaseExtractor):
    """Экстрактор для PostgreSQL базы данных."""

    async def extract(self) -> pd.DataFrame:
        """Выполняет SQL запрос и возвращает результат как DataFrame."""
        conn_config = self.config.get_postresql_connection(
            self.data_source.connection_name
        )

        conn = await asyncpg.connect(conn_config.connection_string)
        try:
            rows = await conn.fetch(self.data_source.sql_query)
            return pd.DataFrame([dict(row) for row in rows])
        finally:
            await conn.close()


class RestAPI(BaseExtractor):
    """Экстрактор для REST API."""

    async def extract(self) -> pd.DataFrame:
        """Получает данные по REST API и нормализует JSON."""
        async with httpx.AsyncClient() as client:
            response = await client.get(self.data_source.url)
            data = response.json()
            return pd.json_normalize(data)


class JSONFile(BaseExtractor):
    """Экстрактор для JSON файлов."""

    async def extract(self) -> pd.DataFrame:
        """Загружает JSON файл и нормализует данные."""
        with open(self.data_source.path, "r") as file:
            data = json.load(file)
            return pd.json_normalize(data)


def get_extractor(data_source: Any, config: AppConfig) -> BaseExtractor:
    """Фабричная функция для создания экстрактора по типу источника данных."""
    extractors = {
        "csv_file": CSVFile,
        "csv_url": CSVUrl,
        "excel_file": ExcelFile,
        "postgresql": PostgreSQL,
        "rest_api": RestAPI,
        "json_file": JSONFile,
    }
    return extractors[data_source.type](data_source, config)
