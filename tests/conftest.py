import pandas as pd
import pytest

from src.config.pipelines import Pipeline


@pytest.fixture
def df_basic():
    return pd.DataFrame(
        {
            "a": [1, 2, 2, 1000],
            "b": ["1", "2", "3", "4"],
            "c": ["x", "y", None, "z"],
            "y": [10, 20, 30, 40],
        }
    )


@pytest.fixture
def df_time():
    return pd.DataFrame(
        {
            "t": pd.date_range("2024-01-01", periods=5, freq="D"),
            "y": [1, 2, 3, 4, 5],
        }
    )


@pytest.fixture
def pipeline_regression():
    return Pipeline(
        name="p",
        data_source={"type": "csv_file", "path": "dummy.csv"},
        model={"type": "regression", "target": "y", "test_size": 0.5},
    )


@pytest.fixture
def pipeline_forecasting():
    return Pipeline(
        name="pf",
        data_source={"type": "csv_file", "path": "dummy.csv"},
        model={"type": "forecasting", "target": "y", "test_size": 0.5},
        time_period=2,
    )
