import pandas as pd

from src.analyzers import MakeDefaultStats, MakeTimeSeriesStats
from src.config.pipelines import Pipeline


def make_pipe(model_type="regression", target="y", is_ts=False, period=None):
    pipe = Pipeline(
        name="p",
        data_source={"type": "csv_file", "path": "dummy.csv"},
        model={"type": model_type, "target": target, "test_size": 0.5},
        time_period=period,
    )
    pipe.meta.is_timeseries = is_ts
    return pipe


def test_make_default_stats():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    p = make_pipe()
    MakeDefaultStats().analyze(df, p)
    assert p.meta.stats is not None
    assert p.meta.stats.mean["x"] == 2


def test_make_timeseries_stats():
    df = pd.DataFrame({"y": [1, 2, 3, 4, 5, 6]})
    p = make_pipe(is_ts=True, period=2)
    MakeTimeSeriesStats().analyze(df, p)
    assert p.meta.timeseries_stats is not None
