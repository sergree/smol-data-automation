"""
Модуль конфигурации приложения.

Содержит классы для загрузки и валидации конфигурационных файлов.
"""

from typing import Type

import yaml
from pydantic import BaseModel

from src.config.connections import (
    BICustomAPIConnection,
    ConnectionsConfig,
    PostgreSQLConnection,
    SMTPConnection,
)
from src.config.pipelines import PipelinesConfig
from src.exceptions import ConnectionNotFound


class AppConfig:
    """Основной класс конфигурации приложения."""

    def __init__(self, connections_yaml_path: str, pipelines_yaml_path: str):
        """Загружает конфигурации соединений и пайплайнов из YAML файлов."""
        self.connections = self.read_yaml(
            connections_yaml_path, ConnectionsConfig
        ).connections
        self.pipelines = self.read_yaml(pipelines_yaml_path, PipelinesConfig).pipelines

    @staticmethod
    def read_yaml(yaml_path: str, validation_schema: Type[BaseModel]):
        """Читает и валидирует YAML файл с помощью Pydantic схемы."""
        with open(yaml_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        return validation_schema(**data)

    def get_connection(
        self, name: str, connection_type: Type[BaseModel] | None = None
    ) -> SMTPConnection | PostgreSQLConnection | BICustomAPIConnection:
        """Получает соединение по имени и типу."""
        for connection in self.connections:
            if connection_type:
                if connection.name == name and isinstance(connection, connection_type):
                    return connection
            else:
                if connection.name == name:
                    return connection
        raise ConnectionNotFound(name)

    def get_smtp_connection(self, name: str) -> SMTPConnection:
        """Получает SMTP соединение по имени."""
        return self.get_connection(name, connection_type=SMTPConnection)

    def get_postresql_connection(self, name: str) -> PostgreSQLConnection:
        """Получает PostgreSQL соединение по имени."""
        return self.get_connection(name, connection_type=PostgreSQLConnection)

    def get_bi_custom_api_connection(self, name: str) -> BICustomAPIConnection:
        """Получает BI API соединение по имени."""
        return self.get_connection(name, connection_type=BICustomAPIConnection)
