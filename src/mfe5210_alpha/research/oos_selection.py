from __future__ import annotations

import math

import pandas as pd


ROBUST_SCORE_COLUMNS = [
    "ls_ann_sharpe",
    "rank_ic_ir",
    "ic_ir",
    "monthly_win_rate",
    "recent_ls_sharpe",
]


def _ann_ret(series: pd.Series) -> float:
    clean = series.dropna()
    return float(clean.mean() * 243) if len(clean) else 0.0


def _ann_sharpe(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return 0.0
    std = clean.std(ddof=0)
    if pd.isna(std) or std == 0:
        return 0.0
    return float(clean.mean() / std * math.sqrt(243))


def _information_ratio(series: pd.Series) -> float:
    return _ann_sharpe(series)


def _monthly_win_rate(series: pd.Series, dates: pd.Series) -> float:
    clean = pd.Series(series.to_numpy(), index=pd.to_datetime(dates)).dropna()
    if clean.empty:
        return 0.0
    monthly = clean.resample("ME").sum()
    if monthly.empty:
        return 0.0
    return float((monthly > 0).mean())


def _max_drawdown(series: pd.Series) -> float:
    clean = series.fillna(0.0)
    if clean.empty:
        return 0.0
    nav = (1.0 + clean).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


def _metric_value(summary_row: pd.Series, raw_name: str, aligned_name: str) -> float:
    value = summary_row.get(raw_name, summary_row.get(aligned_name, 0.0))
    if pd.isna(value):
        return 0.0
    return float(value)


def determine_factor_sign(summary_row: pd.Series) -> int | None:
    rank_ic = _metric_value(summary_row, "raw_rank_ic_mean", "rank_ic_mean")
    spread = _metric_value(summary_row, "raw_ls_ann_ret", "ls_ann_ret")
    if rank_ic > 0 and spread > 0:
        return 1
    if rank_ic < 0 and spread < 0:
        return -1
    return None


def compute_equal_weight_robust_score(summary: pd.DataFrame) -> pd.DataFrame:
    scored = summary.copy()
    if "factor_sign" not in scored.columns:
        scored["factor_sign"] = scored.apply(determine_factor_sign, axis=1)
    for column in ROBUST_SCORE_COLUMNS:
        if column not in scored.columns:
            scored[column] = 0.0
        numeric = pd.to_numeric(scored[column], errors="coerce").fillna(0.0)
        std = numeric.std(ddof=0)
        scored[f"{column}_z"] = 0.0 if pd.isna(std) or std == 0 else (numeric - numeric.mean()) / std
    if "factor_max_drawdown" not in scored.columns:
        scored["factor_max_drawdown"] = 0.0
    drawdown_penalty = (
        pd.to_numeric(scored["factor_max_drawdown"], errors="coerce").fillna(0.0).abs()
    )
    z_cols = [f"{column}_z" for column in ROBUST_SCORE_COLUMNS]
    scored["robust_score"] = scored[z_cols].mean(axis=1) - drawdown_penalty
    return scored


def summarize_window(signal_diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if signal_diagnostics.empty:
        return compute_equal_weight_robust_score(pd.DataFrame(rows))

    for factor_name, group in signal_diagnostics.groupby("factor_name", sort=True):
        ordered = group.sort_values("date").reset_index(drop=True)
        raw_row = {
            "factor_name": factor_name,
            "raw_factor_ann_ret": _ann_ret(ordered["factor_return"]),
            "raw_factor_ann_sharpe": _ann_sharpe(ordered["factor_return"]),
            "raw_ls_ann_ret": _ann_ret(ordered["long_short_return"]),
            "raw_ls_ann_sharpe": _ann_sharpe(ordered["long_short_return"]),
            "raw_ic_mean": float(ordered["ic"].dropna().mean())
            if ordered["ic"].notna().any()
            else 0.0,
            "raw_rank_ic_mean": float(ordered["rank_ic"].dropna().mean())
            if ordered["rank_ic"].notna().any()
            else 0.0,
            "raw_ic_ir": _information_ratio(ordered["ic"]),
            "raw_rank_ic_ir": _information_ratio(ordered["rank_ic"]),
        }
        sign = determine_factor_sign(pd.Series(raw_row))
        multiplier = float(sign if sign is not None else 0)
        aligned_factor = ordered["factor_return"] * multiplier
        aligned_ls = ordered["long_short_return"] * multiplier
        row = {
            **raw_row,
            "factor_sign": sign,
            "factor_ann_ret": _ann_ret(aligned_factor),
            "factor_ann_sharpe": _ann_sharpe(aligned_factor),
            "ls_ann_ret": _ann_ret(aligned_ls),
            "ls_ann_sharpe": _ann_sharpe(aligned_ls),
            "ic_mean": raw_row["raw_ic_mean"] * multiplier,
            "rank_ic_mean": raw_row["raw_rank_ic_mean"] * multiplier,
            "ic_ir": raw_row["raw_ic_ir"] * multiplier,
            "rank_ic_ir": raw_row["raw_rank_ic_ir"] * multiplier,
            "monthly_win_rate": _monthly_win_rate(aligned_ls, ordered["date"]),
            "recent_ls_sharpe": _ann_sharpe(aligned_ls.tail(252)),
            "factor_max_drawdown": _max_drawdown(aligned_ls),
        }
        rows.append(row)

    return compute_equal_weight_robust_score(pd.DataFrame(rows))


def select_factors_for_window(
    summary: pd.DataFrame,
    corr: pd.DataFrame,
    max_abs_corr: float = 0.5,
) -> tuple[list[str], pd.DataFrame]:
    if summary.empty:
        return [], summary.assign(selection_stage=pd.Series(dtype=object))

    details = summary.copy()
    details["selection_stage"] = "dropped_unstable_sign"
    eligible = details.loc[details["factor_sign"].notna()].sort_values(
        ["robust_score", "factor_name"], ascending=[False, True]
    )
    selected: list[str] = []
    for index, row in eligible.iterrows():
        factor_name = str(row["factor_name"])
        is_diversifying = all(
            abs(float(corr.loc[factor_name, chosen])) <= max_abs_corr for chosen in selected
        )
        if is_diversifying:
            selected.append(factor_name)
            details.loc[index, "selection_stage"] = "selected"
        else:
            details.loc[index, "selection_stage"] = "correlation_blocked"
    return selected, details.reset_index(drop=True)


def _composite_ls_sharpe(
    daily_signal_diagnostics: pd.DataFrame,
    factor_names: list[str],
) -> float:
    pivot = (
        daily_signal_diagnostics.pivot(
            index="date", columns="factor_name", values="aligned_long_short_return"
        )
        .reindex(columns=factor_names)
        .dropna(how="all")
        .fillna(0.0)
    )
    if pivot.empty:
        return 0.0
    return _ann_sharpe(pivot.mean(axis=1))


def select_factors_dynamically(
    summary: pd.DataFrame,
    corr: pd.DataFrame,
    daily_signal_diagnostics: pd.DataFrame,
    min_ls_sharpe: float = 0.8,
    min_rank_icir: float = 1.0,
    max_abs_corr: float = 0.5,
    min_incremental_sharpe: float = 0.03,
    max_factors: int = 12,
) -> tuple[list[str], pd.DataFrame]:
    if summary.empty:
        return [], summary.assign(selection_stage=pd.Series(dtype=object))

    details = summary.copy()
    details["selection_stage"] = "not_evaluated"
    details["incremental_train_sharpe"] = 0.0
    ordered = details.sort_values(["robust_score", "factor_name"], ascending=[False, True])
    selected: list[str] = []
    current_sharpe = 0.0

    for index, row in ordered.iterrows():
        factor_name = str(row["factor_name"])
        if pd.isna(row.get("factor_sign")):
            details.loc[index, "selection_stage"] = "dropped_unstable_sign"
            continue
        if (
            float(row.get("ls_ann_sharpe", 0.0)) < min_ls_sharpe
            or float(row.get("rank_ic_ir", 0.0)) < min_rank_icir
        ):
            details.loc[index, "selection_stage"] = "quality_blocked"
            continue
        is_diversifying = all(
            abs(float(corr.loc[factor_name, chosen])) <= max_abs_corr for chosen in selected
        )
        if not is_diversifying:
            details.loc[index, "selection_stage"] = "correlation_blocked"
            continue

        candidate = [*selected, factor_name]
        candidate_sharpe = _composite_ls_sharpe(daily_signal_diagnostics, candidate)
        improvement = candidate_sharpe - current_sharpe
        details.loc[index, "incremental_train_sharpe"] = improvement
        if selected and improvement < min_incremental_sharpe:
            details.loc[index, "selection_stage"] = "increment_blocked"
            continue

        selected.append(factor_name)
        current_sharpe = candidate_sharpe
        details.loc[index, "selection_stage"] = "selected"
        if len(selected) >= max_factors:
            break
    return selected, details.reset_index(drop=True)
