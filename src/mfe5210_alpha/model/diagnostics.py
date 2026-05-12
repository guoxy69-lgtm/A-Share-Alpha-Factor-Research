from __future__ import annotations

import pandas as pd


def compute_quantile_returns(
    score_frame: pd.DataFrame,
    returns: pd.DataFrame,
    quantiles: int = 5,
) -> pd.DataFrame:
    merged = score_frame.merge(returns, on=["date", "security"], how="inner").dropna(
        subset=["score", "fwd_ret_1d"]
    )
    rows = []
    for date, group in merged.groupby("date", sort=True):
        ranked = group.copy()
        ranked["quantile"] = (
            pd.qcut(
                ranked["score"].rank(method="first"),
                quantiles,
                labels=False,
                duplicates="drop",
            )
            + 1
        )
        row = {"date": date}
        quantile_returns = ranked.groupby("quantile")["fwd_ret_1d"].mean()
        for quantile, value in quantile_returns.items():
            row[f"q{int(quantile)}"] = float(value)
        row["q5_minus_q1"] = float(row.get("q5", 0.0) - row.get("q1", 0.0))
        rows.append(row)
    return pd.DataFrame(rows)
