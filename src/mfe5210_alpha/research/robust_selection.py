from __future__ import annotations

import math

import numpy as np
import pandas as pd


BASE_SCORE_COLUMNS = [
    "ann_sharpe",
    "ls_ann_sharpe",
    "rank_ic_ir",
    "ic_ir",
    "ls_ann_ret",
    "monthly_win_rate",
]


def _annualized_sharpe(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty:
        return 0.0
    std = values.std(ddof=0)
    if std == 0 or pd.isna(std):
        return 0.0
    return float(values.mean() / std * math.sqrt(243))


def _max_drawdown(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if values.empty:
        return 0.0
    nav = (1.0 + values).cumprod()
    drawdown = nav / nav.cummax() - 1.0
    return float(drawdown.min())


def _daily_diagnostics(ls_daily: pd.DataFrame, recent_window: int) -> pd.DataFrame:
    rows = []
    if ls_daily is None or ls_daily.empty:
        return pd.DataFrame(
            columns=["factor_name", "recent_ls_sharpe", "factor_max_drawdown"]
        )
    daily = ls_daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    value_col = (
        "aligned_long_short_return"
        if "aligned_long_short_return" in daily.columns
        else "long_short_return"
    )
    for factor_name, group in daily.sort_values("date").groupby("factor_name"):
        returns = group[value_col]
        recent_returns = returns.tail(recent_window)
        rows.append(
            {
                "factor_name": factor_name,
                "recent_ls_sharpe": _annualized_sharpe(recent_returns),
                "factor_max_drawdown": _max_drawdown(returns),
            }
        )
    return pd.DataFrame(rows)


def compute_robust_factor_score(
    summary: pd.DataFrame,
    ls_daily: pd.DataFrame | None = None,
    recent_window: int = 504,
) -> pd.DataFrame:
    scored = summary.copy()
    for column in BASE_SCORE_COLUMNS:
        if column not in scored.columns:
            scored[column] = 0.0

    diagnostics = _daily_diagnostics(ls_daily, recent_window)
    if not diagnostics.empty:
        scored = scored.merge(diagnostics, on="factor_name", how="left")
    for column in ["recent_ls_sharpe", "factor_max_drawdown"]:
        if column not in scored.columns:
            scored[column] = 0.0
    scored[["recent_ls_sharpe", "factor_max_drawdown"]] = scored[
        ["recent_ls_sharpe", "factor_max_drawdown"]
    ].fillna(0.0)

    scored["aligned_ann_sharpe"] = scored["ann_sharpe"].abs()
    scored["aligned_ls_ann_sharpe"] = scored["ls_ann_sharpe"].abs()
    scored["aligned_rank_ic_ir"] = scored["rank_ic_ir"].abs()
    scored["aligned_ic_ir"] = scored["ic_ir"].abs()
    scored["aligned_ls_ann_ret"] = scored["ls_ann_ret"].abs()
    drawdown_penalty = scored["factor_max_drawdown"].abs()

    scored["robust_score"] = (
        0.28 * scored["aligned_ls_ann_sharpe"]
        + 0.20 * scored["aligned_rank_ic_ir"]
        + 0.14 * scored["aligned_ic_ir"]
        + 0.12 * scored["aligned_ann_sharpe"]
        + 0.08 * scored["aligned_ls_ann_ret"]
        + 0.08 * scored["monthly_win_rate"]
        + 0.15 * scored["recent_ls_sharpe"].clip(lower=-5.0, upper=5.0)
        - 0.10 * drawdown_penalty
    )
    return scored


def assign_correlation_clusters(
    corr: pd.DataFrame, cluster_abs_corr: float = 0.7
) -> pd.DataFrame:
    factors = list(corr.index)
    seen: set[str] = set()
    rows = []
    cluster_id = 0
    for factor in factors:
        if factor in seen:
            continue
        stack = [factor]
        component = []
        seen.add(factor)
        while stack:
            current = stack.pop()
            component.append(current)
            related = corr.index[
                corr.loc[current].replace([np.inf, -np.inf], np.nan).abs()
                >= cluster_abs_corr
            ].tolist()
            for candidate in related:
                if candidate not in seen:
                    seen.add(candidate)
                    stack.append(candidate)
        for member in sorted(component):
            rows.append({"factor_name": member, "cluster_id": cluster_id})
        cluster_id += 1
    return pd.DataFrame(rows)


def _mark_cluster_ranks(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    out["cluster_rank"] = (
        out.sort_values(["cluster_id", "robust_score"], ascending=[True, False])
        .groupby("cluster_id")
        .cumcount()
        + 1
    )
    return out


def select_production_factors(
    summary: pd.DataFrame,
    corr: pd.DataFrame,
    ls_daily: pd.DataFrame | None = None,
    max_abs_corr: float = 0.5,
    cluster_abs_corr: float = 0.7,
    recent_window: int = 504,
) -> tuple[list[str], pd.DataFrame]:
    scored = compute_robust_factor_score(summary, ls_daily, recent_window=recent_window)
    clusters = assign_correlation_clusters(corr, cluster_abs_corr=cluster_abs_corr)
    details = scored.merge(clusters, on="factor_name", how="left")
    details["cluster_id"] = details["cluster_id"].fillna(-1).astype(int)
    details = _mark_cluster_ranks(details)

    candidate_order = details.sort_values(
        ["cluster_rank", "robust_score"], ascending=[True, False]
    )["factor_name"].tolist()
    selected: list[str] = []
    selection_stage: dict[str, str] = {}

    for candidate in candidate_order:
        row = details.loc[details["factor_name"] == candidate].iloc[0]
        if int(row["cluster_rank"]) > 1:
            selection_stage[candidate] = "cluster_redundant"
            continue
        if all(abs(float(corr.loc[candidate, chosen])) <= max_abs_corr for chosen in selected):
            selected.append(candidate)
            selection_stage[candidate] = "selected"
        else:
            selection_stage[candidate] = "correlation_blocked"

    details["selected"] = details["factor_name"].isin(selected)
    details["selection_stage"] = details["factor_name"].map(selection_stage).fillna(
        "not_evaluated"
    )
    return selected, details.sort_values("robust_score", ascending=False).reset_index(drop=True)
