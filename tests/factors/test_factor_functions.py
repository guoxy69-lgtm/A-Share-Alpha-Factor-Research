import pandas as pd

from mfe5210_alpha.factors.momentum import factor_return_5d
from mfe5210_alpha.factors.wq_like import factor_wq_alpha_005


def test_factor_return_5d_creates_a_named_column():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=6),
            "security": ["sz.000001"] * 6,
            "close": [10, 10.5, 10.8, 11.0, 10.9, 11.2],
            "open": [9.9, 10.4, 10.7, 10.9, 10.8, 11.0],
            "volume": [100, 110, 115, 120, 118, 130],
        }
    )
    out = factor_return_5d(frame)
    assert "return_5d" in out.columns


def test_factor_wq_alpha_005_creates_a_named_column():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=6),
            "security": ["sz.000001"] * 6,
            "open": [9.9, 10.4, 10.7, 10.9, 10.8, 11.0],
            "close": [10, 10.5, 10.8, 11.0, 10.9, 11.2],
            "high": [10.1, 10.6, 10.9, 11.1, 11.0, 11.3],
            "low": [9.8, 10.3, 10.6, 10.8, 10.7, 10.9],
            "volume": [100, 110, 115, 120, 118, 130],
        }
    )
    out = factor_wq_alpha_005(frame)
    assert "wq_alpha_005" in out.columns
