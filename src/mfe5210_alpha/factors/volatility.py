from __future__ import annotations

import pandas as pd

from mfe5210_alpha.factors.base import output_factor, sort_panel, ts_std


def _make_volatility_factor(window: int):
    def _factor(frame: pd.DataFrame) -> pd.DataFrame:
        out = sort_panel(frame)
        ret = out.groupby("security")["close"].pct_change()
        values = ret.groupby(out["security"]).rolling(window, min_periods=2).std().reset_index(level=0, drop=True)
        return output_factor(out, f"volatility_{window}d", values)

    return _factor


def _make_range_vol_factor(window: int):
    def _factor(frame: pd.DataFrame) -> pd.DataFrame:
        out = sort_panel(frame)
        intraday = (out["high"] - out["low"]) / out["close"].replace(0, pd.NA)
        values = intraday.groupby(out["security"]).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        return output_factor(out, f"range_vol_{window}d", values)

    return _factor


FACTOR_FUNCTIONS: dict[str, callable] = {}
for _window in [5, 10, 20, 40, 60, 120]:
    FACTOR_FUNCTIONS[f"volatility_{_window}d"] = _make_volatility_factor(_window)
    FACTOR_FUNCTIONS[f"range_vol_{_window}d"] = _make_range_vol_factor(_window)
