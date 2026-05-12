from __future__ import annotations

import math

import pandas as pd

from mfe5210_alpha.research.preprocess import winsorize_by_date, zscore_by_date


def _ann_ret(series: pd.Series) -> float:
    return float(series.mean() * 243) if len(series) else 0.0


def _ann_sharpe(series: pd.Series) -> float:
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return 0.0
    return float(series.mean() / std * math.sqrt(243))


def _daily_long_short_returns(work: pd.DataFrame) -> pd.Series:
    subset = work[["date", "security", "factor_value", "fwd_ret_1d"]].copy()
    counts = subset.groupby("date")["security"].transform("size")
    bucket_size = (counts // 5).clip(lower=1)
    subset["rank_desc"] = subset.groupby("date")["factor_value"].rank(
        method="first", ascending=False
    )
    subset["rank_asc"] = subset.groupby("date")["factor_value"].rank(
        method="first", ascending=True
    )
    top = subset.loc[subset["rank_desc"] <= bucket_size].groupby("date")["fwd_ret_1d"].mean()
    bottom = subset.loc[subset["rank_asc"] <= bucket_size].groupby("date")["fwd_ret_1d"].mean()
    return top.sub(bottom, fill_value=0.0)


def _grouped_corr(work: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
    corr = work.groupby("date")[[x_col, y_col]].corr()
    if corr.empty:
        return pd.Series(dtype=float)
    series = corr.loc[(slice(None), x_col), y_col]
    series.index = series.index.droplevel(-1)
    return series.fillna(0.0)


def _daily_ic(work: pd.DataFrame) -> pd.Series:
    return _grouped_corr(work[["date", "factor_value", "fwd_ret_1d"]], "factor_value", "fwd_ret_1d")


def _daily_rank_ic(work: pd.DataFrame) -> pd.Series:
    ranked = work[["date", "factor_value", "fwd_ret_1d"]].copy()
    ranked["factor_rank"] = ranked.groupby("date")["factor_value"].rank(method="average")
    ranked["return_rank"] = ranked.groupby("date")["fwd_ret_1d"].rank(method="average")
    return _grouped_corr(ranked[["date", "factor_rank", "return_rank"]], "factor_rank", "return_rank")


def _information_ratio(series: pd.Series) -> float:
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return 0.0
    return float(series.mean() / std * math.sqrt(243))


def _monthly_win_rate(aligned_ls_daily: pd.Series) -> float:
    if aligned_ls_daily.empty:
        return 0.0
    monthly = aligned_ls_daily.resample("ME").sum()
    if monthly.empty:
        return 0.0
    return float((monthly > 0).mean())


def prepare_factor_frame(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.dropna(subset=["factor_value", "fwd_ret_1d"]).copy()
    if work.empty:
        return work
    work["factor_value"] = winsorize_by_date(work, "factor_value")
    work["factor_value"] = zscore_by_date(work, "factor_value")
    work["factor_return"] = work["factor_value"] * work["fwd_ret_1d"]
    return work


def evaluate_factor(frame: pd.DataFrame, factor_name: str) -> dict[str, object]:
    work = prepare_factor_frame(frame)
    if work.empty:
        return {
            "factor_name": factor_name,
            "ann_ret": 0.0,
            "ann_sharpe": 0.0,
            "ls_ann_ret": 0.0,
            "ls_ann_sharpe": 0.0,
            "ic_mean": 0.0,
            "rank_ic_mean": 0.0,
            "ic_ir": 0.0,
            "rank_ic_ir": 0.0,
            "monthly_win_rate": 0.0,
            "rev_flag": False,
        }

    daily = work.groupby("date")["factor_return"].mean()
    ls_daily = _daily_long_short_returns(work)
    daily_ic = _daily_ic(work)
    daily_rank_ic = _daily_rank_ic(work)
    rev_flag = bool(ls_daily.mean() < 0)
    aligned_ls_daily = ls_daily * (-1.0 if rev_flag else 1.0)

    return {
        "factor_name": factor_name,
        "ann_ret": _ann_ret(daily),
        "ann_sharpe": _ann_sharpe(daily),
        "ls_ann_ret": _ann_ret(ls_daily),
        "ls_ann_sharpe": _ann_sharpe(ls_daily),
        "ic_mean": float(daily_ic.mean()) if len(daily_ic) else 0.0,
        "rank_ic_mean": float(daily_rank_ic.mean()) if len(daily_rank_ic) else 0.0,
        "ic_ir": _information_ratio(daily_ic),
        "rank_ic_ir": _information_ratio(daily_rank_ic),
        "monthly_win_rate": _monthly_win_rate(aligned_ls_daily),
        "rev_flag": rev_flag,
    }


def compute_daily_factor_series(
    frame: pd.DataFrame, factor_name: str, rev_flag: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = prepare_factor_frame(frame)
    if work.empty:
        return (
            pd.DataFrame(columns=["date", "factor_return", "factor_name", "aligned_factor_return"]),
            pd.DataFrame(columns=["date", "long_short_return", "factor_name", "aligned_long_short_return"]),
        )

    factor_daily = (
        work.groupby("date")["factor_return"].mean().rename("factor_return").reset_index()
    )
    factor_daily["factor_name"] = factor_name
    factor_daily["aligned_factor_return"] = factor_daily["factor_return"] * (
        -1.0 if rev_flag else 1.0
    )

    ls_daily = _daily_long_short_returns(work).rename("long_short_return").reset_index()
    ls_daily["factor_name"] = factor_name
    ls_daily["aligned_long_short_return"] = ls_daily["long_short_return"] * (
        -1.0 if rev_flag else 1.0
    )
    return factor_daily, ls_daily
