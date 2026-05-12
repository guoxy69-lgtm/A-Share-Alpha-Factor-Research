from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from mfe5210_alpha.experiments.exposures import prepare_weighted_exposures
from mfe5210_alpha.model.backtest import (
    build_buffered_long_short_positions,
    compute_portfolio_return,
    summarize_backtest,
)
from mfe5210_alpha.model.composite import build_weighted_score
from mfe5210_alpha.model.dynamic import (
    build_rolling_factor_weights,
    estimate_fama_macbeth_premia,
)
from mfe5210_alpha.research.oos_selection import (
    select_factors_for_window,
    summarize_window,
)


PERFORMANCE_COLUMNS = ["date", "portfolio_return", "cum_nav", "drawdown"]
POSITION_COLUMNS = ["date", "security", "weight"]


def build_fixed_split_windows(
    sample_start: str,
    sample_end: str,
    train_end: str,
    test_start: str,
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    return {
        "train": (pd.Timestamp(sample_start), pd.Timestamp(train_end)),
        "test": (pd.Timestamp(test_start), pd.Timestamp(sample_end)),
    }


def _mask_between(
    frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series:
    return (frame["date"] >= start) & (frame["date"] <= end)


def _factor_correlation(train_diag: pd.DataFrame) -> pd.DataFrame:
    if train_diag.empty:
        return pd.DataFrame()
    return (
        train_diag.pivot(
            index="date", columns="factor_name", values="long_short_return"
        )
        .corr()
        .fillna(0.0)
    )


def _freeze_weights_for_test_dates(
    weights: pd.DataFrame,
    test_dates: pd.Series,
    factor_names: list[str],
) -> pd.DataFrame:
    columns = ["date", *factor_names]
    unique_dates = (
        pd.Series(pd.to_datetime(test_dates).dropna().unique()).sort_values().reset_index(drop=True)
    )
    if unique_dates.empty or not factor_names:
        return pd.DataFrame(columns=columns)

    if weights.empty:
        frozen = pd.Series(1.0 / len(factor_names), index=factor_names)
    else:
        frozen = weights.sort_values("date")[factor_names].tail(1).iloc[0]

    out = pd.DataFrame({"date": unique_dates})
    for factor_name in factor_names:
        out[factor_name] = float(frozen[factor_name])
    return out[columns]


def run_fixed_split_experiment(
    panel: pd.DataFrame,
    signal_diagnostics: pd.DataFrame,
    factor_root: Path,
    cfg: Any,
    top_frac: float = 0.10,
    exit_buffer_frac: float = 0.15,
    max_factors: int = 8,
) -> dict[str, pd.DataFrame]:
    panel = panel.copy()
    signal_diagnostics = signal_diagnostics.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    signal_diagnostics["date"] = pd.to_datetime(signal_diagnostics["date"])

    windows = build_fixed_split_windows(
        sample_start=str(panel["date"].min().date()),
        sample_end=str(panel["date"].max().date()),
        train_end=cfg.fixed_split.train_end,
        test_start=cfg.fixed_split.test_start,
    )

    train_diag = signal_diagnostics.loc[
        _mask_between(signal_diagnostics, *windows["train"])
    ].copy()
    train_summary = summarize_window(train_diag)
    selected_names, selection_details = select_factors_for_window(
        train_summary, _factor_correlation(train_diag), max_abs_corr=0.5
    )
    selected_names = selected_names[:max_factors]
    selection_details = selection_details.loc[
        selection_details["factor_name"].isin(selected_names)
    ].reset_index(drop=True)
    factor_signs = (
        train_summary.set_index("factor_name")["factor_sign"].dropna().astype(int).to_dict()
        if not train_summary.empty
        else {}
    )

    returns = panel[["date", "security", "fwd_ret_1d"]].drop_duplicates()
    exposures = prepare_weighted_exposures(panel, factor_root, selected_names, factor_signs)
    train_exposures = exposures.loc[_mask_between(exposures, *windows["train"])].copy()
    premia = estimate_fama_macbeth_premia(train_exposures, returns, selected_names, min_obs=100)
    rolling_weights = build_rolling_factor_weights(
        premia, selected_names, lookback=252, min_periods=60, max_weight=0.35
    )

    test_exposures = exposures.loc[_mask_between(exposures, *windows["test"])].copy()
    fixed_weights = _freeze_weights_for_test_dates(
        rolling_weights, test_exposures["date"], selected_names
    )
    test_score = build_weighted_score(test_exposures, fixed_weights, selected_names)
    test_positions = build_buffered_long_short_positions(
        test_score, top_frac=top_frac, exit_buffer_frac=exit_buffer_frac
    )[POSITION_COLUMNS]
    test_performance = compute_portfolio_return(test_positions, returns)[PERFORMANCE_COLUMNS]
    test_summary = summarize_backtest(test_performance, test_positions)
    test_summary.insert(0, "model_name", "fixed_train_test_weighted")

    return {
        "performance": test_performance,
        "positions": test_positions,
        "summary": test_summary,
        "selection_details": selection_details,
        "weights": fixed_weights,
    }
