from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from mfe5210_alpha.experiments.exposures import prepare_weighted_exposures
from mfe5210_alpha.experiments.fixed_split import (
    PERFORMANCE_COLUMNS,
    POSITION_COLUMNS,
    _factor_correlation,
    _mask_between,
)
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


WINDOW_COLUMNS = [
    "window_train_start",
    "window_train_end",
    "window_trade_start",
    "window_trade_end",
]


def build_walk_forward_windows(
    sample_start: str,
    sample_end: str,
    lookback_years: int,
    holdout_years: int,
    step_years: int = 1,
) -> list[dict[str, pd.Timestamp]]:
    start = pd.Timestamp(sample_start)
    end = pd.Timestamp(sample_end)
    first_trade_year = start.year + lookback_years
    windows: list[dict[str, pd.Timestamp]] = []
    for trade_year in range(first_trade_year, end.year + 1, step_years):
        train_start = pd.Timestamp(f"{trade_year - lookback_years}-01-01")
        train_end = pd.Timestamp(f"{trade_year - 1}-12-31")
        trade_start = pd.Timestamp(f"{trade_year}-01-01")
        trade_end = min(pd.Timestamp(f"{trade_year + holdout_years - 1}-12-31"), end)
        if trade_start > end:
            break
        windows.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "trade_start": trade_start,
                "trade_end": trade_end,
            }
        )
    return windows


def _annotate_window(frame: pd.DataFrame, window: dict[str, pd.Timestamp]) -> pd.DataFrame:
    annotated = frame.copy()
    annotated["window_train_start"] = window["train_start"]
    annotated["window_train_end"] = window["train_end"]
    annotated["window_trade_start"] = window["trade_start"]
    annotated["window_trade_end"] = window["trade_end"]
    return annotated


def _empty_selection_details() -> pd.DataFrame:
    return pd.DataFrame(columns=["factor_name", "selection_stage", *WINDOW_COLUMNS])


def _empty_weights() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", *WINDOW_COLUMNS])


def _recompute_global_nav(performance: pd.DataFrame) -> pd.DataFrame:
    if performance.empty:
        return pd.DataFrame(columns=[*PERFORMANCE_COLUMNS, *WINDOW_COLUMNS])
    ordered = performance.sort_values("date").reset_index(drop=True)
    ordered["cum_nav"] = (1.0 + ordered["portfolio_return"]).cumprod()
    ordered["drawdown"] = ordered["cum_nav"] / ordered["cum_nav"].cummax() - 1.0
    return ordered


def run_walk_forward_experiment(
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

    windows = build_walk_forward_windows(
        sample_start=str(panel["date"].min().date()),
        sample_end=str(panel["date"].max().date()),
        lookback_years=cfg.walk_forward.lookback_years,
        holdout_years=cfg.walk_forward.holdout_years,
        step_years=getattr(cfg.walk_forward, "step_years", 1),
    )
    returns = panel[["date", "security", "fwd_ret_1d"]].drop_duplicates()

    performance_frames: list[pd.DataFrame] = []
    position_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []
    weight_frames: list[pd.DataFrame] = []

    for window in windows:
        train_diag = signal_diagnostics.loc[
            _mask_between(signal_diagnostics, window["train_start"], window["train_end"])
        ].copy()
        window_summary = summarize_window(train_diag)
        selected_names, selection_details = select_factors_for_window(
            window_summary, _factor_correlation(train_diag), max_abs_corr=0.5
        )
        selected_names = selected_names[:max_factors]
        selection_frames.append(_annotate_window(selection_details, window))
        if not selected_names:
            continue

        factor_signs = (
            window_summary.set_index("factor_name")["factor_sign"]
            .dropna()
            .astype(int)
            .to_dict()
        )
        exposures = prepare_weighted_exposures(panel, factor_root, selected_names, factor_signs)
        premia = estimate_fama_macbeth_premia(
            exposures.loc[exposures["date"] <= window["trade_end"]].copy(),
            returns,
            selected_names,
            min_obs=100,
        )
        weights = build_rolling_factor_weights(
            premia,
            selected_names,
            lookback=252,
            min_periods=60,
            max_weight=0.35,
        )
        weights = weights.loc[
            _mask_between(weights, window["trade_start"], window["trade_end"])
        ].copy()
        if weights.empty:
            continue

        trade_exposures = exposures.loc[
            _mask_between(exposures, window["trade_start"], window["trade_end"])
        ].copy()
        score = build_weighted_score(trade_exposures, weights, selected_names)
        positions = build_buffered_long_short_positions(
            score, top_frac=top_frac, exit_buffer_frac=exit_buffer_frac
        )
        if positions.empty:
            continue
        positions = _annotate_window(positions[POSITION_COLUMNS], window)
        position_frames.append(positions)
        weight_frames.append(_annotate_window(weights, window))

        performance = compute_portfolio_return(positions[POSITION_COLUMNS], returns)
        if performance.empty:
            continue
        performance_frames.append(_annotate_window(performance[PERFORMANCE_COLUMNS], window))

    performance = _recompute_global_nav(
        pd.concat(performance_frames, ignore_index=True)
        if performance_frames
        else pd.DataFrame(columns=[*PERFORMANCE_COLUMNS, *WINDOW_COLUMNS])
    )
    positions = (
        pd.concat(position_frames, ignore_index=True)
        if position_frames
        else pd.DataFrame(columns=[*POSITION_COLUMNS, *WINDOW_COLUMNS])
    )
    selection_details = (
        pd.concat(selection_frames, ignore_index=True)
        if selection_frames
        else _empty_selection_details()
    )
    weights = (
        pd.concat(weight_frames, ignore_index=True)
        if weight_frames
        else _empty_weights()
    )

    summary = summarize_backtest(performance, positions[POSITION_COLUMNS] if not positions.empty else positions)
    summary.insert(0, "model_name", "rolling_train_test_weighted")
    return {
        "performance": performance,
        "positions": positions,
        "summary": summary,
        "selection_details": selection_details,
        "weights": weights,
    }
