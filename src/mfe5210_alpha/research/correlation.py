from __future__ import annotations

import pandas as pd


def build_factor_correlation(daily_factor_returns: pd.DataFrame) -> pd.DataFrame:
    wide = daily_factor_returns.pivot(
        index="date", columns="factor_name", values="aligned_factor_return"
    )
    return wide.corr()


def compute_research_score(summary: pd.DataFrame) -> pd.DataFrame:
    scored = summary.copy()
    for column in [
        "ann_sharpe",
        "ls_ann_sharpe",
        "ann_ret",
        "ls_ann_ret",
        "rank_ic_ir",
        "ic_ir",
        "monthly_win_rate",
    ]:
        if column not in scored.columns:
            scored[column] = 0.0
    scored["aligned_ann_sharpe"] = scored["ann_sharpe"].abs()
    scored["aligned_ls_ann_sharpe"] = scored["ls_ann_sharpe"].abs()
    scored["aligned_ann_ret"] = scored["ann_ret"].abs()
    scored["aligned_ls_ann_ret"] = scored["ls_ann_ret"].abs()
    scored["aligned_rank_ic_ir"] = scored["rank_ic_ir"].abs()
    scored["aligned_ic_ir"] = scored["ic_ir"].abs()
    scored["research_score"] = (
        0.35 * scored["aligned_ls_ann_sharpe"]
        + 0.20 * scored["aligned_rank_ic_ir"]
        + 0.15 * scored["aligned_ic_ir"]
        + 0.15 * scored["aligned_ann_sharpe"]
        + 0.10 * scored["aligned_ls_ann_ret"]
        + 0.05 * scored["monthly_win_rate"]
    )
    return scored


def greedy_select_factors(
    summary: pd.DataFrame, corr: pd.DataFrame, max_abs_corr: float = 0.5
) -> list[str]:
    ordered = summary.sort_values("research_score", ascending=False)["factor_name"].tolist()
    selected: list[str] = []
    for candidate in ordered:
        if not selected:
            selected.append(candidate)
            continue
        if all(abs(float(corr.loc[candidate, chosen])) <= max_abs_corr for chosen in selected):
            selected.append(candidate)
    return selected
