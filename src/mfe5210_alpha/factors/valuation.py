from __future__ import annotations

import pandas as pd

from mfe5210_alpha.factors.base import cs_zscore, output_factor, sort_panel


def _make_level_factor(field: str):
    def _factor(frame: pd.DataFrame) -> pd.DataFrame:
        out = sort_panel(frame)
        return output_factor(out, f"{field}_level", out[field])

    return _factor


def _make_zscore_factor(field: str):
    def _factor(frame: pd.DataFrame) -> pd.DataFrame:
        out = sort_panel(frame)
        values = cs_zscore(out, field)
        return output_factor(out, f"{field}_zscore", values)

    return _factor


FACTOR_FUNCTIONS: dict[str, callable] = {}
for _field in ["peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]:
    FACTOR_FUNCTIONS[f"{_field}_level"] = _make_level_factor(_field)
    FACTOR_FUNCTIONS[f"{_field}_zscore"] = _make_zscore_factor(_field)
