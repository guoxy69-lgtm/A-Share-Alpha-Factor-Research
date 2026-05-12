from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FactorDefinition:
    name: str
    family: str
    description: str
    formula_text: str
    source_quote: str


def ts_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def ts_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=2).std()


def cs_zscore(frame: pd.DataFrame, value_col: str) -> pd.Series:
    grouped = frame.groupby("date")[value_col]
    return grouped.transform(lambda x: (x - x.mean()) / x.std(ddof=0))


def sort_panel(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["security", "date"]).copy()


def output_factor(frame: pd.DataFrame, name: str, values: pd.Series) -> pd.DataFrame:
    out = frame[["date", "security"]].copy()
    out[name] = values
    return out
