from __future__ import annotations

import pandas as pd

from mfe5210_alpha.data.download_daily import A_SHARE_PREFIXES


NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "amount",
    "turn",
    "pctChg",
    "peTTM",
    "pbMRQ",
    "psTTM",
    "pcfNcfTTM",
    "isST",
]


def build_research_panel(
    raw_prices: pd.DataFrame, allowed_securities: set[str] | None = None
) -> pd.DataFrame:
    frame = raw_prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[frame["code"].astype(str).str.startswith(A_SHARE_PREFIXES)].copy()
    if allowed_securities is not None:
        frame = frame[frame["code"].isin(allowed_securities)].copy()
    frame = frame.rename(columns={"code": "security"})

    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.sort_values(["security", "date"]).reset_index(drop=True)
    frame["ret_1d"] = frame.groupby("security")["close"].pct_change()
    frame["fwd_ret_1d"] = frame.groupby("security")["ret_1d"].shift(-1)
    frame["adv20"] = (
        frame.groupby("security")["amount"]
        .rolling(20, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    frame["turnover_proxy"] = frame["turn"].fillna(0.0)
    return frame
