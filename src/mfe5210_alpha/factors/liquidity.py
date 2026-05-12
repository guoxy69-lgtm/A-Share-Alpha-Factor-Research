from __future__ import annotations

import pandas as pd

from mfe5210_alpha.factors.base import output_factor, sort_panel


def _make_amount_mean_factor(window: int):
    def _factor(frame: pd.DataFrame) -> pd.DataFrame:
        out = sort_panel(frame)
        values = out.groupby("security")["amount"].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        return output_factor(out, f"amount_mean_{window}d", values)

    return _factor


def _make_turnover_mean_factor(window: int):
    def _factor(frame: pd.DataFrame) -> pd.DataFrame:
        out = sort_panel(frame)
        values = out.groupby("security")["turn"].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        return output_factor(out, f"turnover_mean_{window}d", values)

    return _factor


FACTOR_FUNCTIONS: dict[str, callable] = {}
for _window in [5, 10, 20, 40, 60, 120]:
    FACTOR_FUNCTIONS[f"amount_mean_{_window}d"] = _make_amount_mean_factor(_window)
    FACTOR_FUNCTIONS[f"turnover_mean_{_window}d"] = _make_turnover_mean_factor(_window)
