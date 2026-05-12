from __future__ import annotations

import pandas as pd


def winsorize_by_date(
    frame: pd.DataFrame, value_col: str, lower: float = 0.01, upper: float = 0.99
) -> pd.Series:
    group_size = frame.groupby("date")[value_col].transform("size")
    if int(group_size.max()) < 20:
        return frame[value_col]

    return frame.groupby("date")[value_col].transform(
        lambda x: x.clip(x.quantile(lower), x.quantile(upper))
    )


def zscore_by_date(frame: pd.DataFrame, value_col: str) -> pd.Series:
    def _transform(x: pd.Series) -> pd.Series:
        std = x.std(ddof=0)
        if pd.isna(std) or std == 0:
            return pd.Series(0.0, index=x.index)
        return (x - x.mean()) / std

    return frame.groupby("date")[value_col].transform(_transform)
