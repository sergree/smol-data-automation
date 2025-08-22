"""
Модуль пользовательских исключений.

Содержит специфичные для приложения исключения.
"""


class ConnectionNotFound(Exception):
    """Исключение, возникающее когда соединение не найдено."""

    def __init__(self, connection_name: str):
        super().__init__(f"Соединение `{connection_name}` не найдено")
