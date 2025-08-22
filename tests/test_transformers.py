import pandas as pd

from src.config.pipelines import Pipeline, TimeSeriesStats
from src.transformers import (
    DropColumnsManual,
    DropNotNumeric,
    Encode,
    PrepareForecastingFeatures,
    Scale,
    SetTimeColumn,
)


def base_pipeline(target="y"):
    return Pipeline(
        name="p",
        data_source={"type": "csv_file", "path": "dummy.csv"},
        model={"type": "regression", "target": target},
    )


def test_drop_and_time(df_time):
    pipe = base_pipeline()
    pipe.transform.pick_columns = ["t", "y"]
    df = DropColumnsManual().transform(df_time.copy(), pipe)
    pipe.time_column = "t"
    df2 = SetTimeColumn().transform(df, pipe)
    assert "y" in df2.columns and pipe.meta.is_timeseries


def test_encode_and_drop_non_numeric():
    df = pd.DataFrame(
        {"cat": ["a", "b", "a"], "color": ["r", "g", "r"], "y": [1, 2, 3]}
    )
    pipe = base_pipeline()
    pipe.transform.encode.label = ["cat"]
    pipe.transform.encode.onehot = ["color"]
    df1 = Encode().transform(df.copy(), pipe)
    df2 = DropNotNumeric().transform(df1, pipe)
    assert all(col in df2.columns for col in ["cat", "color_g", "color_r", "y"])
    assert df2.select_dtypes(exclude="number").empty


def test_scale_excludes_target_and_no_scale():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [10.0, 20.0], "y": [5.0, 7.0]})
    pipe = base_pipeline(target="y")
    pipe.transform.no_scale = ["a"]
    df2 = Scale().transform(df.copy(), pipe)
    assert df2["a"].equals(df["a"])
    assert df2["y"].equals(df["y"])
    assert not df2["b"].equals(df["b"])


def test_prepare_forecasting_features(df_time):
    pipe = base_pipeline(target="y")
    pipe.model.type = "forecasting"
    pipe.time_column = "t"
    df = SetTimeColumn().transform(df_time.copy(), pipe)
    pipe.meta.timeseries_stats = TimeSeriesStats(
        trend=pd.Series([0, 0, 0, 0, 0], index=df.index),
        seasonal=pd.Series([0, 0, 0, 0, 0], index=df.index),
    )
    df2 = PrepareForecastingFeatures().transform(df, pipe)
    assert all(
        col in df2.columns
        for col in ["y_trend", "y_seasonal", "y_lag_1", "y_lag_2", "y_lag_3", "y_ma_3"]
    )
    assert len(df2) == 2
