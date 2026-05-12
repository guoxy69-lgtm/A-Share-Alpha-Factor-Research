from __future__ import annotations

import math

import numpy as np
import pandas as pd


DEFAULT_COMMISSION_BPS = 3.6
DEFAULT_SLIPPAGE_BPS = 0.0


def build_long_short_positions(score_frame: pd.DataFrame, top_frac: float = 0.1) -> pd.DataFrame:
    work = score_frame.copy()
    bucket_size = max(1, int(len(work) * top_frac))
    work["rank_desc"] = work["score"].rank(method="first", ascending=False)
    work["rank_asc"] = work["score"].rank(method="first", ascending=True)
    work["weight"] = 0.0
    work.loc[work["rank_desc"] <= bucket_size, "weight"] = 1.0 / bucket_size
    work.loc[work["rank_asc"] <= bucket_size, "weight"] = -1.0 / bucket_size
    columns = ["security", "weight"]
    if "date" in work.columns:
        columns.insert(0, "date")
    return work.loc[work["weight"] != 0, columns]


def _buffer_size(universe_size: int, frac: float, floor_size: int) -> int:
    return max(floor_size, int(math.ceil(universe_size * frac)))


def _fill_target_positions(
    ordered: list[str], kept: list[str], bucket_size: int, blocked: set[str] | None = None
) -> list[str]:
    blocked = blocked or set()
    target = [security for security in kept if security not in blocked]
    for security in ordered:
        if security in blocked or security in target:
            continue
        target.append(security)
        if len(target) >= bucket_size:
            break
    return target[:bucket_size]


def build_buffered_long_short_positions(
    score_frame: pd.DataFrame,
    top_frac: float = 0.10,
    exit_buffer_frac: float = 0.15,
) -> pd.DataFrame:
    if score_frame.empty:
        return pd.DataFrame(columns=["date", "security", "weight"])

    position_frames = []
    current_positions = pd.DataFrame(columns=["security", "weight"])
    daily_scores = score_frame.dropna(subset=["score"]).sort_values(["date", "security"])
    for date, group in daily_scores.groupby("date", sort=True):
        bucket_size = max(1, int(len(group) * top_frac))
        exit_size = _buffer_size(len(group), exit_buffer_frac, bucket_size)
        ranked_long = group.sort_values(["score", "security"], ascending=[False, True]).reset_index(
            drop=True
        )
        ranked_short = group.sort_values(["score", "security"], ascending=[True, True]).reset_index(
            drop=True
        )

        if current_positions.empty:
            current_positions = build_long_short_positions(
                group[["security", "score"]], top_frac=top_frac
            )[["security", "weight"]]
        else:
            prev_long = set(current_positions.loc[current_positions["weight"] > 0, "security"])
            prev_short = set(current_positions.loc[current_positions["weight"] < 0, "security"])
            long_buffer = ranked_long.head(exit_size)["security"].tolist()
            short_buffer = ranked_short.head(exit_size)["security"].tolist()
            kept_long = [security for security in long_buffer if security in prev_long]
            kept_short = [security for security in short_buffer if security in prev_short]
            long_target = _fill_target_positions(
                ranked_long["security"].tolist(), kept_long, bucket_size
            )
            short_target = _fill_target_positions(
                ranked_short["security"].tolist(), kept_short, bucket_size, blocked=set(long_target)
            )
            current_positions = pd.concat(
                [
                    pd.DataFrame({"security": long_target, "weight": 1.0 / bucket_size}),
                    pd.DataFrame({"security": short_target, "weight": -1.0 / bucket_size}),
                ],
                ignore_index=True,
            )

        dated_positions = current_positions.copy()
        dated_positions.insert(0, "date", date)
        position_frames.append(dated_positions)

    if not position_frames:
        return pd.DataFrame(columns=["date", "security", "weight"])
    return pd.concat(position_frames, ignore_index=True)


def build_rebalanced_long_short_positions(
    score_frame: pd.DataFrame,
    top_frac: float = 0.1,
    rebalance_every: int = 1,
) -> pd.DataFrame:
    if score_frame.empty:
        return pd.DataFrame(columns=["date", "security", "weight"])

    rebalance_every = max(1, int(rebalance_every))
    position_frames = []
    current_positions = pd.DataFrame(columns=["security", "weight"])
    daily_scores = score_frame.dropna(subset=["score"]).sort_values(["date", "security"])
    for day_no, (date, group) in enumerate(daily_scores.groupby("date", sort=True)):
        if day_no % rebalance_every == 0 or current_positions.empty:
            current_positions = build_long_short_positions(
                group[["security", "score"]], top_frac=top_frac
            )
        if current_positions.empty:
            continue
        dated_positions = current_positions.copy()
        dated_positions.insert(0, "date", date)
        position_frames.append(dated_positions)

    if not position_frames:
        return pd.DataFrame(columns=["date", "security", "weight"])
    return pd.concat(position_frames, ignore_index=True)


def _normalize_long_short_weights(positions: pd.DataFrame) -> pd.DataFrame:
    out = positions.copy()
    long_sum = out.loc[out["weight"] > 0, "weight"].sum()
    short_sum = -out.loc[out["weight"] < 0, "weight"].sum()
    if long_sum > 0:
        out.loc[out["weight"] > 0, "weight"] = out.loc[out["weight"] > 0, "weight"] / long_sum
    if short_sum > 0:
        out.loc[out["weight"] < 0, "weight"] = out.loc[out["weight"] < 0, "weight"] / short_sum
    return out


def smooth_positions(target_positions: pd.DataFrame, alpha: float = 0.65) -> pd.DataFrame:
    if target_positions.empty:
        return pd.DataFrame(columns=["date", "security", "weight"])

    alpha = float(np.clip(alpha, 0.0, 1.0))
    target = (
        target_positions.pivot(index="date", columns="security", values="weight")
        .fillna(0.0)
        .sort_index()
    )
    previous = None
    position_frames = []
    for date, target_row in target.iterrows():
        smoothed = target_row.copy() if previous is None else (1.0 - alpha) * previous + alpha * target_row
        day_positions = smoothed.loc[smoothed.abs() > 1e-8].rename("weight").reset_index()
        day_positions = _normalize_long_short_weights(day_positions)
        day_positions.insert(0, "date", date)
        position_frames.append(day_positions)
        previous = day_positions.set_index("security")["weight"].reindex(target.columns).fillna(0.0)

    return pd.concat(position_frames, ignore_index=True)


def build_long_only_positions(score_frame: pd.DataFrame, top_frac: float = 0.20) -> pd.DataFrame:
    if score_frame.empty:
        return pd.DataFrame(columns=["date", "security", "weight"])

    frames = []
    for _, group in score_frame.dropna(subset=["score"]).groupby("date", sort=True):
        bucket_size = max(1, int(len(group) * top_frac))
        chosen = group.sort_values(["score", "security"], ascending=[False, True]).head(
            bucket_size
        )
        frames.append(chosen[["date", "security"]].assign(weight=1.0 / bucket_size))
    if not frames:
        return pd.DataFrame(columns=["date", "security", "weight"])
    return pd.concat(frames, ignore_index=True)


def _pivot_positions(positions: pd.DataFrame) -> pd.DataFrame:
    return positions.pivot(index="date", columns="security", values="weight").fillna(0.0).sort_index()


def _one_way_turnover(positions: pd.DataFrame) -> pd.Series:
    pivoted = _pivot_positions(positions)
    previous = pivoted.shift(1).fillna(0.0)
    return pivoted.sub(previous).abs().sum(axis=1).div(2.0)


def compute_portfolio_return(
    positions: pd.DataFrame,
    returns: pd.DataFrame,
    commission_bps: float = DEFAULT_COMMISSION_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> pd.DataFrame:
    merged = positions.merge(returns, on=["date", "security"], how="left")
    daily = (
        merged[["date", "weight", "fwd_ret_1d"]]
        .groupby("date")
        .apply(lambda x: (x["weight"] * x["fwd_ret_1d"]).sum(), include_groups=False)
    )
    out = daily.rename("gross_portfolio_return").reset_index()
    turnover = _one_way_turnover(positions).rename("one_way_turnover").reset_index()
    out = out.merge(turnover, on="date", how="left")
    out["holding_pnl"] = out["gross_portfolio_return"]
    out["trading_pnl"] = 0.0
    out["commission"] = out["one_way_turnover"] * (commission_bps / 10000.0)
    out["slippage"] = out["one_way_turnover"] * (slippage_bps / 10000.0)
    out["total_pnl"] = out["holding_pnl"] + out["trading_pnl"]
    out["net_portfolio_return"] = out["total_pnl"] - out["commission"] - out["slippage"]
    out["portfolio_return"] = out["net_portfolio_return"]
    out["cum_nav"] = (1 + out["portfolio_return"]).cumprod()
    out["drawdown"] = out["cum_nav"] / out["cum_nav"].cummax() - 1.0
    return out


def summarize_backtest(performance: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    daily = (
        performance["net_portfolio_return"]
        if "net_portfolio_return" in performance.columns
        else performance["portfolio_return"]
    )
    std = daily.std(ddof=0)
    turnover = (
        performance["one_way_turnover"]
        if "one_way_turnover" in performance.columns
        else _one_way_turnover(positions).reindex(performance["date"]).fillna(0.0)
    )
    return pd.DataFrame(
        [
            {
                "annualized_return": float(daily.mean() * 243) if len(daily) else 0.0,
                "annualized_sharpe": float(daily.mean() / std * math.sqrt(243))
                if std not in (0, None) and not pd.isna(std)
                else 0.0,
                "max_drawdown": float(performance["drawdown"].min()) if len(performance) else 0.0,
                "average_one_way_turnover": float(turnover.mean()) if len(turnover) else 0.0,
            }
        ]
    )
