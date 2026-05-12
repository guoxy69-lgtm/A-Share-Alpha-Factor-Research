from __future__ import annotations

import math

import numpy as np
import pandas as pd


def estimate_fama_macbeth_premia(
    exposures: pd.DataFrame,
    returns: pd.DataFrame,
    factor_names: list[str],
    min_obs: int = 50,
) -> pd.DataFrame:
    merged = exposures.merge(returns, on=["date", "security"], how="inner")
    rows: list[dict[str, object]] = []
    for date, group in merged.groupby("date", sort=True):
        complete = group[[*factor_names, "fwd_ret_1d"]].replace([np.inf, -np.inf], np.nan).dropna()
        row: dict[str, object] = {"date": date}
        if len(complete) < min_obs:
            row.update({name: 0.0 for name in factor_names})
            rows.append(row)
            continue
        x = complete[factor_names].to_numpy(dtype=float)
        y = complete["fwd_ret_1d"].to_numpy(dtype=float)
        x = np.column_stack([np.ones(len(complete)), x])
        beta = np.linalg.lstsq(x, y, rcond=None)[0][1:]
        row.update({name: float(value) for name, value in zip(factor_names, beta)})
        rows.append(row)
    return pd.DataFrame(rows)


def _cap_weights(weights: pd.Series, max_weight: float) -> pd.Series:
    capped = weights.clip(lower=0.0, upper=max_weight)
    for _ in range(len(capped) + 1):
        deficit = 1.0 - capped.sum()
        if abs(deficit) < 1e-12:
            break
        room = max_weight - capped
        room = room[room > 1e-12]
        if room.empty:
            break
        increment = room / room.sum() * deficit
        capped.loc[room.index] = (capped.loc[room.index] + increment).clip(upper=max_weight)
    total = capped.sum()
    if total <= 0:
        return pd.Series(1.0 / len(weights), index=weights.index)
    return capped / total


def build_rolling_factor_weights(
    premia: pd.DataFrame,
    factor_names: list[str],
    lookback: int = 252,
    min_periods: int = 60,
    max_weight: float = 0.35,
) -> pd.DataFrame:
    ordered = premia.sort_values("date").set_index("date")
    rolling_mean = ordered[factor_names].rolling(lookback, min_periods=min_periods).mean().shift(1)
    rolling_std = ordered[factor_names].rolling(lookback, min_periods=min_periods).std(ddof=0).shift(1)
    strength = rolling_mean.div(rolling_std.where(rolling_std.abs() > 1e-12))
    strength = strength.where(~strength.isna(), rolling_mean).clip(lower=0.0)

    rows = []
    equal = pd.Series(1.0 / len(factor_names), index=factor_names)
    for date, row in strength.iterrows():
        raw = row.fillna(0.0)
        weights = equal if raw.sum() <= 0 else raw / raw.sum()
        weights = _cap_weights(weights, max_weight=max_weight)
        out = {"date": date}
        out.update({name: float(weights[name]) for name in factor_names})
        rows.append(out)
    return pd.DataFrame(rows)


def apply_factor_weights(
    exposures: pd.DataFrame, weights: pd.DataFrame, factor_names: list[str]
) -> pd.DataFrame:
    merged = exposures.merge(weights, on="date", how="left", suffixes=("", "_weight"))
    score = pd.Series(0.0, index=merged.index)
    for name in factor_names:
        score = score + merged[name] * merged[name + "_weight"].fillna(1.0 / len(factor_names))
    return merged[["date", "security"]].assign(score=score)


def apply_volatility_target(
    performance: pd.DataFrame,
    target_vol: float = 0.25,
    lookback: int = 20,
    max_leverage: float = 1.0,
) -> pd.DataFrame:
    out = performance.sort_values("date").copy()
    realized_vol = out["portfolio_return"].rolling(lookback, min_periods=lookback).std(ddof=0)
    realized_vol = realized_vol.mul(math.sqrt(243)).shift(1)
    leverage = target_vol / realized_vol.where(realized_vol > 1e-12)
    out["leverage"] = leverage.clip(lower=0.0, upper=max_leverage).fillna(max_leverage)
    out["scaled_portfolio_return"] = out["portfolio_return"] * out["leverage"]
    out["scaled_cum_nav"] = (1.0 + out["scaled_portfolio_return"]).cumprod()
    out["scaled_drawdown"] = out["scaled_cum_nav"] / out["scaled_cum_nav"].cummax() - 1.0
    return out
