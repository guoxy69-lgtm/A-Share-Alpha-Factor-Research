from __future__ import annotations

import pandas as pd


def summarize_sample_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    work = panel.copy()
    work["date"] = pd.to_datetime(work["date"])
    return pd.DataFrame(
        [
            {
                "start_date": work["date"].min().strftime("%Y-%m-%d"),
                "end_date": work["date"].max().strftime("%Y-%m-%d"),
                "trading_days": int(work["date"].nunique()),
                "security_count": int(work["security"].nunique()),
                "row_count": int(len(work)),
                "missing_turn_ratio": float(work["turn"].isna().mean())
                if "turn" in work.columns and len(work)
                else 1.0,
            }
        ]
    )
