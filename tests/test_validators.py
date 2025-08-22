from types import SimpleNamespace

import pandas as pd

from src.config.pipelines import Pipeline
from src.validators import FullDuplicates, NaNs, Outliers, WrongTypes


def make_pipeline(**kwargs):
    p = SimpleNamespace()
    p.meta = SimpleNamespace(is_timeseries=False)
    t = SimpleNamespace(
        missing_strategy=kwargs.get("missing_strategy", "drop"),
        outlier_strategy=kwargs.get("outlier_strategy", "iqr"),
    )
    p.transform = t
    return p


def test_full_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
    v = FullDuplicates().validate(df, make_pipeline())
    df2 = v.fix(df, make_pipeline())
    assert len(df2) == 2


def test_wrong_types():
    df = pd.DataFrame({"x": ["1", "2", "3"], "y": [1, 2, 3]})

    pipe = Pipeline(
        name="p",
        data_source={"type": "csv_file", "path": "dummy.csv"},
        model={"type": "regression", "target": "y"},
    )
    v = WrongTypes().validate(df, pipe)
    df2 = v.fix(df, pipe)
    assert df2["x"].dtype.kind in "iu"


def test_nans_mean():
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", None, "z"]})

    pipe = Pipeline(
        name="p",
        data_source={"type": "csv_file", "path": "dummy.csv"},
        model={"type": "regression", "target": "a"},
    )
    pipe.transform.missing_strategy = "mean"
    v = NaNs().validate(df.copy(), pipe)
    df2 = v.fix(df.copy(), pipe)
    assert df2.isna().sum().sum() == 0


def test_outliers_iqr():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 1000.0]})

    pipe = Pipeline(
        name="p",
        data_source={"type": "csv_file", "path": "dummy.csv"},
        model={"type": "regression", "target": "a"},
    )
    v = Outliers().validate(df, pipe)
    df2 = v.fix(df, pipe)
    assert len(df2) == 3
