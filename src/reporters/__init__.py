"""
Модуль для создания отчетов и отправки результатов.

Содержит классы для генерации отчетов в различных форматах и отправки их по email.
"""

from src.reporters.composite_file_email import CompositeFileEmailReporter
from src.reporters.database import PostgreSQLReporter
from src.reporters.mock_bi_custom_api import MockBICustomAPIReporter

__all__ = [
    "CompositeFileEmailReporter",
    "MockBICustomAPIReporter",
    "PostgreSQLReporter",
]
