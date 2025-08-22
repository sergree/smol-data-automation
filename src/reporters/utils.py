"""
Утилиты для репортеров.

Содержит вспомогательные функции для сериализации данных.
"""

from io import BytesIO
from typing import Any

import joblib
import pandas as pd

from src.config.pipelines import Stats, TimeSeriesStats


def serialize_model(model: Any) -> bytes:
    """Сериализует ML модель в байты с помощью joblib."""
    buffer = BytesIO()
    joblib.dump(model, buffer, compress=3)
    return buffer.getvalue()


def convert_timestamps(obj: Any) -> Any:
    """Конвертирует pandas Timestamp в ISO строки для JSON сериализации."""
    if isinstance(obj, dict):
        return {
            (k.isoformat() if isinstance(k, pd.Timestamp) else k): convert_timestamps(v)
            for k, v in obj.items()
            if not pd.isna(convert_timestamps(v))
        }
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    return obj


def serialize_series_stats(stats_obj, fields: list[str]) -> dict:
    """Сериализует статистики pandas Series в словарь."""
    if stats_obj is None:
        return None

    return {
        field: convert_timestamps(getattr(stats_obj, field).to_dict())
        if hasattr(getattr(stats_obj, field, None), "to_dict")
        else None
        for field in fields
    }


def serialize_stats(stats: Stats) -> dict:
    """Сериализует объект Stats в словарь."""
    return serialize_series_stats(stats, ["mean", "median", "mode", "std"])


def serialize_timeseries_stats(ts_stats: TimeSeriesStats) -> dict:
    """Сериализует объект TimeSeriesStats в словарь."""
    return serialize_series_stats(ts_stats, ["trend", "seasonal"])
