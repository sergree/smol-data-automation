"""
Модуль конфигурации соединений.

Содержит Pydantic модели для различных типов соединений.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class SMTPConnection(BaseModel):
    """Конфигурация SMTP соединения для отправки email."""

    name: str
    type: Literal["smtp"]
    host: str
    port: int = Field(..., ge=1, le=65535)
    username: str
    password: str


class PostgreSQLConnection(BaseModel):
    """Конфигурация PostgreSQL соединения."""

    name: str
    type: Literal["postgresql"]
    connection_string: str


class BICustomAPIConnection(BaseModel):
    """Конфигурация соединения с BI системой через API."""

    name: str
    type: Literal["bi_custom_api"]
    api_url: str
    token: str


class ConnectionsConfig(BaseModel):
    """Конфигурация всех соединений приложения."""

    connections: list[
        Annotated[
            SMTPConnection | PostgreSQLConnection | BICustomAPIConnection,
            Field(discriminator="type"),
        ]
    ]
