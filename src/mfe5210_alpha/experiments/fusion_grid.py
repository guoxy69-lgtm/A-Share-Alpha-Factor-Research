from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from mfe5210_alpha.experiments.exposures import prepare_weighted_exposures
from mfe5210_alpha.experiments.fixed_split import (
    PERFORMANCE_COLUMNS,
    POSITION_COLUMNS,
    _factor_correlation,
    _mask_between,
    build_fixed_split_windows,
)
from mfe5210_alpha.experiments.walk_forward import (
    WINDOW_COLUMNS,
    _annotate_window,
    _recompute_global_nav,
    build_walk_forward_windows,
)
from mfe5210_alpha.model.backtest import (
    build_buffered_long_short_positions,
    build_long_only_positions,
    compute_portfolio_return,
    summarize_backtest,
)
from mfe5210_alpha.model.composite import build_weighted_score
from mfe5210_alpha.model.diagnostics import compute_quantile_returns
from mfe5210_alpha.model.fusion import build_factor_weights
from mfe5210_alpha.research.oos_selection import (
    select_factors_dynamically,
    summarize_window,
)


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    route: str
    fusion_method: str
    top_frac: float
    exit_buffer_frac: float
    neutralize: bool
    neutralization_method: str
    use_industry: bool
    portfolio_type: str


def build_method_specs() -> list[ExperimentSpec]:
    methods = [
        "equal_weight",
        "ls_sharpe_weight",
        "rank_icir_weight",
        "ic_mean_weight",
        "robust_score_weight",
        "max_ir",
    ]
    specs: list[ExperimentSpec] = []
    for method in methods:
        specs.append(
            ExperimentSpec(
                name=f"fixed_{method.replace('_weight', '')}_ls20_neutral_ols",
                route="fixed",
                fusion_method=method,
                top_frac=0.20,
                exit_buffer_frac=0.25,
                neutralize=True,
                neutralization_method="ols",
                use_industry=True,
                portfolio_type="long_short",
            )
        )
    specs.append(
        ExperimentSpec(
            name="fixed_equal_ls20_raw",
            route="fixed",
            fusion_method="equal_weight",
            top_frac=0.20,
            exit_buffer_frac=0.25,
            neutralize=False,
            neutralization_method="ols",
            use_industry=False,
            portfolio_type="long_short",
        )
    )
    return specs


def _constant_weights_for_dates(
    dates: pd.Series,
    weights: pd.Series,
    model_name: str,
) -> pd.DataFrame:
    unique_dates = (
        pd.Series(pd.to_datetime(dates).dropna().unique()).sort_values().reset_index(drop=True)
    )
    out = pd.DataFrame({"date": unique_dates})
    for factor_name, value in weights.items():
        out[str(factor_name)] = float(value)
    out.insert(0, "model_name", model_name)
    return out


def _empty_result(model_name: str) -> dict[str, pd.DataFrame]:
    summary = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "annualized_return": 0.0,
                "annualized_sharpe": 0.0,
                "max_drawdown": 0.0,
                "average_one_way_turnover": 0.0,
            }
        ]
    )
    return {
        "performance": pd.DataFrame(columns=["model_name", *PERFORMANCE_COLUMNS]),
        "positions": pd.DataFrame(columns=["model_name", *POSITION_COLUMNS]),
        "summary": summary,
        "weights": pd.DataFrame(columns=["model_name", "date"]),
        "selection_details": pd.DataFrame(
            columns=["model_name", "factor_name", "selection_stage"]
        ),
        "quantiles": pd.DataFrame(columns=["model_name", "date"]),
    }


def _build_positions(score: pd.DataFrame, spec: ExperimentSpec) -> pd.DataFrame:
    if spec.portfolio_type == "long_only":
        return build_long_only_positions(score, top_frac=spec.top_frac)
    return build_buffered_long_short_positions(
        score,
        top_frac=spec.top_frac,
        exit_buffer_frac=spec.exit_buffer_frac,
    )


def _tag_frame(frame: pd.DataFrame, model_name: str) -> pd.DataFrame:
    tagged = frame.copy()
    if "model_name" not in tagged.columns:
        tagged.insert(0, "model_name", model_name)
    return tagged


def _add_aligned_long_short_return(
    diagnostics: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    aligned = diagnostics.copy()
    if aligned.empty:
        aligned["aligned_long_short_return"] = pd.Series(dtype=float)
        return aligned
    signs = summary.set_index("factor_name")["factor_sign"].dropna().astype(int).to_dict()
    aligned["factor_sign"] = aligned["factor_name"].map(signs).fillna(0).astype(float)
    aligned["aligned_long_short_return"] = aligned["long_short_return"] * aligned["factor_sign"]
    return aligned


def run_fixed_fusion_spec(
    panel: pd.DataFrame,
    signal_diagnostics: pd.DataFrame,
    factor_root: Path,
    cfg: Any,
    spec: ExperimentSpec,
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
    train_diag = signal_diagnostics.loc[_mask_between(signal_diagnostics, *windows["train"])].copy()
    train_summary = summarize_window(train_diag)
    aligned_train_diag = _add_aligned_long_short_return(train_diag, train_summary)
    selected_names, selection_details = select_factors_dynamically(
        train_summary,
        _factor_correlation(train_diag),
        aligned_train_diag,
        min_ls_sharpe=0.8,
        min_rank_icir=1.0,
        max_abs_corr=0.5,
        min_incremental_sharpe=0.03,
        max_factors=12,
    )
    selection_details = _tag_frame(selection_details, spec.name)
    if not selected_names:
        result = _empty_result(spec.name)
        result["selection_details"] = selection_details
        return result

    factor_signs = train_summary.set_index("factor_name")["factor_sign"].dropna().astype(int).to_dict()
    returns = panel[["date", "security", "fwd_ret_1d"]].drop_duplicates()
    test_panel = panel.loc[_mask_between(panel, *windows["test"])].copy()
    exposures = prepare_weighted_exposures(
        test_panel,
        factor_root,
        selected_names,
        factor_signs,
        neutralize=spec.neutralize,
        neutralization_method=spec.neutralization_method,
        use_industry=spec.use_industry,
    )
    test_exposures = exposures.loc[_mask_between(exposures, *windows["test"])].copy()
    weights = build_factor_weights(
        method=spec.fusion_method,
        factor_names=selected_names,
        summary=train_summary,
        daily_signal_diagnostics=aligned_train_diag,
        max_weight=0.50,
    )
    weights_frame = _constant_weights_for_dates(test_exposures["date"], weights, spec.name)
    score = build_weighted_score(
        test_exposures, weights_frame.drop(columns=["model_name"]), selected_names
    )
    positions = _build_positions(score, spec)[POSITION_COLUMNS]
    performance = compute_portfolio_return(positions, returns)[PERFORMANCE_COLUMNS]
    summary = summarize_backtest(performance, positions)
    summary.insert(0, "model_name", spec.name)
    quantiles = compute_quantile_returns(score, returns)
    return {
        "performance": _tag_frame(performance, spec.name),
        "positions": _tag_frame(positions, spec.name),
        "summary": summary,
        "weights": weights_frame,
        "selection_details": selection_details,
        "quantiles": _tag_frame(quantiles, spec.name),
    }


def run_walk_forward_fusion_spec(
    panel: pd.DataFrame,
    signal_diagnostics: pd.DataFrame,
    factor_root: Path,
    cfg: Any,
    spec: ExperimentSpec,
) -> dict[str, pd.DataFrame]:
    panel = panel.copy()
    signal_diagnostics = signal_diagnostics.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    signal_diagnostics["date"] = pd.to_datetime(signal_diagnostics["date"])
    returns = panel[["date", "security", "fwd_ret_1d"]].drop_duplicates()
    windows = build_walk_forward_windows(
        sample_start=str(panel["date"].min().date()),
        sample_end=str(panel["date"].max().date()),
        lookback_years=cfg.walk_forward.lookback_years,
        holdout_years=cfg.walk_forward.holdout_years,
        step_years=getattr(cfg.walk_forward, "step_years", 1),
    )

    performance_frames: list[pd.DataFrame] = []
    position_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []
    weight_frames: list[pd.DataFrame] = []
    quantile_frames: list[pd.DataFrame] = []
    for window in windows:
        train_diag = signal_diagnostics.loc[
            _mask_between(signal_diagnostics, window["train_start"], window["train_end"])
        ].copy()
        train_summary = summarize_window(train_diag)
        aligned_train_diag = _add_aligned_long_short_return(train_diag, train_summary)
        selected_names, selection_details = select_factors_dynamically(
            train_summary,
            _factor_correlation(train_diag),
            aligned_train_diag,
            min_ls_sharpe=0.8,
            min_rank_icir=1.0,
            max_abs_corr=0.5,
            min_incremental_sharpe=0.03,
            max_factors=12,
        )
        selection_frames.append(_annotate_window(_tag_frame(selection_details, spec.name), window))
        if not selected_names:
            continue

        factor_signs = (
            train_summary.set_index("factor_name")["factor_sign"].dropna().astype(int).to_dict()
        )
        trade_panel = panel.loc[
            _mask_between(panel, window["trade_start"], window["trade_end"])
        ].copy()
        exposures = prepare_weighted_exposures(
            trade_panel,
            factor_root,
            selected_names,
            factor_signs,
            neutralize=spec.neutralize,
            neutralization_method=spec.neutralization_method,
            use_industry=spec.use_industry,
        )
        trade_exposures = exposures.loc[
            _mask_between(exposures, window["trade_start"], window["trade_end"])
        ].copy()
        if trade_exposures.empty:
            continue
        weights = build_factor_weights(
            method=spec.fusion_method,
            factor_names=selected_names,
            summary=train_summary,
            daily_signal_diagnostics=aligned_train_diag,
            max_weight=0.50,
        )
        weights_frame = _constant_weights_for_dates(trade_exposures["date"], weights, spec.name)
        score = build_weighted_score(
            trade_exposures, weights_frame.drop(columns=["model_name"]), selected_names
        )
        positions = _build_positions(score, spec)
        if positions.empty:
            continue

        performance = compute_portfolio_return(positions[POSITION_COLUMNS], returns)[
            PERFORMANCE_COLUMNS
        ]
        quantiles = compute_quantile_returns(score, returns)
        performance_frames.append(_annotate_window(_tag_frame(performance, spec.name), window))
        position_frames.append(_annotate_window(_tag_frame(positions[POSITION_COLUMNS], spec.name), window))
        weight_frames.append(_annotate_window(weights_frame, window))
        quantile_frames.append(_annotate_window(_tag_frame(quantiles, spec.name), window))

    if performance_frames:
        performance = pd.concat(performance_frames, ignore_index=True).drop(columns=["model_name"])
        performance = _tag_frame(_recompute_global_nav(performance), spec.name)
    else:
        performance = pd.DataFrame(columns=["model_name", *PERFORMANCE_COLUMNS, *WINDOW_COLUMNS])
    positions = (
        pd.concat(position_frames, ignore_index=True)
        if position_frames
        else pd.DataFrame(columns=["model_name", *POSITION_COLUMNS, *WINDOW_COLUMNS])
    )
    weights = (
        pd.concat(weight_frames, ignore_index=True)
        if weight_frames
        else pd.DataFrame(columns=["model_name", "date", *WINDOW_COLUMNS])
    )
    selection_details = (
        pd.concat(selection_frames, ignore_index=True)
        if selection_frames
        else pd.DataFrame(columns=["model_name", "factor_name", "selection_stage", *WINDOW_COLUMNS])
    )
    quantiles = (
        pd.concat(quantile_frames, ignore_index=True)
        if quantile_frames
        else pd.DataFrame(columns=["model_name", "date", *WINDOW_COLUMNS])
    )
    summary = summarize_backtest(performance, positions[POSITION_COLUMNS])
    summary.insert(0, "model_name", spec.name)
    return {
        "performance": performance,
        "positions": positions,
        "summary": summary,
        "weights": weights,
        "selection_details": selection_details,
        "quantiles": quantiles,
    }


def _finalize_walk_forward_result(
    model_name: str,
    frames: dict[str, list[pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    if frames["performance"]:
        performance = pd.concat(frames["performance"], ignore_index=True).drop(
            columns=["model_name"]
        )
        performance = _tag_frame(_recompute_global_nav(performance), model_name)
    else:
        performance = pd.DataFrame(columns=["model_name", *PERFORMANCE_COLUMNS, *WINDOW_COLUMNS])
    positions = (
        pd.concat(frames["positions"], ignore_index=True)
        if frames["positions"]
        else pd.DataFrame(columns=["model_name", *POSITION_COLUMNS, *WINDOW_COLUMNS])
    )
    weights = (
        pd.concat(frames["weights"], ignore_index=True)
        if frames["weights"]
        else pd.DataFrame(columns=["model_name", "date", *WINDOW_COLUMNS])
    )
    selection_details = (
        pd.concat(frames["selection_details"], ignore_index=True)
        if frames["selection_details"]
        else pd.DataFrame(columns=["model_name", "factor_name", "selection_stage", *WINDOW_COLUMNS])
    )
    quantiles = (
        pd.concat(frames["quantiles"], ignore_index=True)
        if frames["quantiles"]
        else pd.DataFrame(columns=["model_name", "date", *WINDOW_COLUMNS])
    )
    summary = summarize_backtest(performance, positions[POSITION_COLUMNS])
    summary.insert(0, "model_name", model_name)
    return {
        "performance": performance,
        "positions": positions,
        "summary": summary,
        "weights": weights,
        "selection_details": selection_details,
        "quantiles": quantiles,
    }


def run_walk_forward_fusion_specs(
    panel: pd.DataFrame,
    signal_diagnostics: pd.DataFrame,
    factor_root: Path,
    cfg: Any,
    specs: list[ExperimentSpec],
) -> list[dict[str, pd.DataFrame]]:
    panel = panel.copy()
    signal_diagnostics = signal_diagnostics.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    signal_diagnostics["date"] = pd.to_datetime(signal_diagnostics["date"])
    returns = panel[["date", "security", "fwd_ret_1d"]].drop_duplicates()
    windows = build_walk_forward_windows(
        sample_start=str(panel["date"].min().date()),
        sample_end=str(panel["date"].max().date()),
        lookback_years=cfg.walk_forward.lookback_years,
        holdout_years=cfg.walk_forward.holdout_years,
        step_years=getattr(cfg.walk_forward, "step_years", 1),
    )
    frames_by_model = {
        spec.name: {
            "performance": [],
            "positions": [],
            "selection_details": [],
            "weights": [],
            "quantiles": [],
        }
        for spec in specs
    }

    for window in windows:
        train_diag = signal_diagnostics.loc[
            _mask_between(signal_diagnostics, window["train_start"], window["train_end"])
        ].copy()
        train_summary = summarize_window(train_diag)
        aligned_train_diag = _add_aligned_long_short_return(train_diag, train_summary)
        selected_names, selection_details = select_factors_dynamically(
            train_summary,
            _factor_correlation(train_diag),
            aligned_train_diag,
            min_ls_sharpe=0.8,
            min_rank_icir=1.0,
            max_abs_corr=0.5,
            min_incremental_sharpe=0.03,
            max_factors=12,
        )
        for spec in specs:
            frames_by_model[spec.name]["selection_details"].append(
                _annotate_window(_tag_frame(selection_details, spec.name), window)
            )
        if not selected_names:
            continue

        factor_signs = (
            train_summary.set_index("factor_name")["factor_sign"].dropna().astype(int).to_dict()
        )
        trade_panel = panel.loc[
            _mask_between(panel, window["trade_start"], window["trade_end"])
        ].copy()
        exposure_cache: dict[tuple[bool, str, bool], pd.DataFrame] = {}

        for spec in specs:
            exposure_key = (spec.neutralize, spec.neutralization_method, spec.use_industry)
            if exposure_key not in exposure_cache:
                exposure_cache[exposure_key] = prepare_weighted_exposures(
                    trade_panel,
                    factor_root,
                    selected_names,
                    factor_signs,
                    neutralize=spec.neutralize,
                    neutralization_method=spec.neutralization_method,
                    use_industry=spec.use_industry,
                )
            trade_exposures = exposure_cache[exposure_key]
            if trade_exposures.empty:
                continue

            weights = build_factor_weights(
                method=spec.fusion_method,
                factor_names=selected_names,
                summary=train_summary,
                daily_signal_diagnostics=aligned_train_diag,
                max_weight=0.50,
            )
            weights_frame = _constant_weights_for_dates(
                trade_exposures["date"], weights, spec.name
            )
            score = build_weighted_score(
                trade_exposures, weights_frame.drop(columns=["model_name"]), selected_names
            )
            positions = _build_positions(score, spec)
            if positions.empty:
                continue

            performance = compute_portfolio_return(positions[POSITION_COLUMNS], returns)[
                PERFORMANCE_COLUMNS
            ]
            quantiles = compute_quantile_returns(score, returns)
            bucket = frames_by_model[spec.name]
            bucket["performance"].append(
                _annotate_window(_tag_frame(performance, spec.name), window)
            )
            bucket["positions"].append(
                _annotate_window(_tag_frame(positions[POSITION_COLUMNS], spec.name), window)
            )
            bucket["weights"].append(_annotate_window(weights_frame, window))
            bucket["quantiles"].append(
                _annotate_window(_tag_frame(quantiles, spec.name), window)
            )

    return [
        _finalize_walk_forward_result(spec.name, frames_by_model[spec.name])
        for spec in specs
    ]
