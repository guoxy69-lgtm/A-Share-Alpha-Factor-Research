from __future__ import annotations

import pandas as pd

from mfe5210_alpha.factors.base import output_factor, sort_panel


def _make_return_factor(window: int):
    def _factor(frame: pd.DataFrame) -> pd.DataFrame:
        out = sort_panel(frame)
        values = out.groupby("security")["close"].pct_change(window)
        return output_factor(out, f"return_{window}d", values)

    return _factor


def _make_reversal_factor(window: int):
    def _factor(frame: pd.DataFrame) -> pd.DataFrame:
        out = sort_panel(frame)
        values = -out.groupby("security")["close"].pct_change(window)
        return output_factor(out, f"reversal_{window}d", values)

    return _factor


def _make_gap_drift_factor(window: int):
    def _factor(frame: pd.DataFrame) -> pd.DataFrame:
        out = sort_panel(frame)
        gap = (out["close"] - out["open"]) / out["open"].replace(0, pd.NA)
        values = gap.groupby(out["security"]).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        return output_factor(out, f"gap_drift_{window}d", values)

    return _factor


factor_return_5d = _make_return_factor(5)


FACTOR_FUNCTIONS: dict[str, callable] = {}
for _window in [5, 10, 20, 60, 120, 240]:
    FACTOR_FUNCTIONS[f"return_{_window}d"] = _make_return_factor(_window)
    FACTOR_FUNCTIONS[f"reversal_{_window}d"] = _make_reversal_factor(_window)
    FACTOR_FUNCTIONS[f"gap_drift_{_window}d"] = _make_gap_drift_factor(_window)
