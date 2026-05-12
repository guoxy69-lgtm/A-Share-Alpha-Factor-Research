import pandas as pd

from mfe5210_alpha.factors.base import cs_zscore, ts_mean


def test_ts_mean_returns_grouped_rolling_mean():
    series = pd.Series([1.0, 2.0, 3.0, 4.0])
    out = ts_mean(series, window=2)
    assert list(out.round(2)) == [1.0, 1.5, 2.5, 3.5]


def test_cs_zscore_standardizes_each_date():
    frame = pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"],
            "value": [1.0, 3.0, 2.0, 4.0],
        }
    )
    out = cs_zscore(frame, "value")
    assert round(float(out.iloc[0]), 6) == -1.0
    assert round(float(out.iloc[1]), 6) == 1.0
