from pathlib import Path

import pandas as pd
import math

from mfe5210_alpha.config import ProjectConfig
from mfe5210_alpha.model.backtest import (
    build_rebalanced_long_short_positions,
    compute_portfolio_return,
    smooth_positions,
    summarize_backtest,
)
from mfe5210_alpha.model.dynamic import (
    apply_factor_weights,
    apply_volatility_target,
    build_rolling_factor_weights,
    estimate_fama_macbeth_premia,
)
from mfe5210_alpha.model.optimizer import build_optimized_long_short_positions
from mfe5210_alpha.research.neutralization import build_style_controls, neutralize_factor_matrix
from mfe5210_alpha.research.preprocess import winsorize_by_date, zscore_by_date


CONTROL_COLUMNS = ["size_proxy", "beta_proxy", "volatility_proxy"]
SMOOTHING_ALPHA = 0.65
COMMISSION_BPS = 3.6
SLIPPAGE_BPS = 0.0


def _prepare_selected_exposures(
    panel: pd.DataFrame, selected: pd.DataFrame, factor_root: Path
) -> tuple[pd.DataFrame, list[str]]:
    controls = build_style_controls(panel)
    base = panel[["date", "security", "fwd_ret_1d"]].drop_duplicates()
    base = base.merge(controls, on=["date", "security"], how="left")
    selected_names = selected["factor_name"].tolist()

    factor_matrix = base.copy()
    for name in selected_names:
        factor = pd.read_parquet(factor_root / f"{name}.parquet")[["date", "security", name]]
        factor_matrix = factor_matrix.merge(factor, on=["date", "security"], how="left")

    neutral = neutralize_factor_matrix(
        factor_matrix,
        factor_cols=selected_names,
        control_cols=CONTROL_COLUMNS,
    )
    exposures = neutral[["date", "security"]].copy()
    for name in selected_names:
        neutral_col = f"{name}_neutral"
        processed_col = f"{name}_processed"
        neutral[processed_col] = winsorize_by_date(neutral, neutral_col)
        neutral[processed_col] = zscore_by_date(neutral, processed_col)
        rev_flag = bool(selected.loc[selected["factor_name"] == name, "rev_flag"].iloc[0])
        exposures[name] = neutral[processed_col] * (-1.0 if rev_flag else 1.0)
    return exposures, selected_names


def _build_positions(
    score: pd.DataFrame,
    top_frac: float = 0.10,
    rebalance_every: int = 1,
) -> pd.DataFrame:
    return build_rebalanced_long_short_positions(
        score[["date", "security", "score"]],
        top_frac=top_frac,
        rebalance_every=rebalance_every,
    )


def _build_risk_frame(panel: pd.DataFrame) -> pd.DataFrame:
    work = panel.sort_values(["security", "date"]).copy()
    ret = pd.to_numeric(work["ret_1d"], errors="coerce")
    work["risk_proxy"] = (
        ret.groupby(work["security"])
        .rolling(20, min_periods=5)
        .std()
        .reset_index(level=0, drop=True)
    )
    work["risk_proxy"] = work.groupby("date")["risk_proxy"].transform(
        lambda x: x.fillna(x.median() if x.notna().any() else 0.02)
    )
    return work[["date", "security", "risk_proxy"]]


def _period_metrics(frame: pd.DataFrame, label: str) -> dict[str, object]:
    returns = frame["scaled_portfolio_return"]
    std = returns.std(ddof=0)
    nav = (1.0 + returns).cumprod()
    drawdown = nav / nav.cummax() - 1.0
    return {
        "period": label,
        "start_date": frame["date"].min(),
        "end_date": frame["date"].max(),
        "trading_days": len(frame),
        "annualized_return": float(returns.mean() * 243) if len(frame) else 0.0,
        "annualized_sharpe": float(returns.mean() / std * math.sqrt(243))
        if std not in (0, None) and not pd.isna(std)
        else 0.0,
        "max_drawdown": float(drawdown.min()) if len(frame) else 0.0,
        "ending_nav": float(nav.iloc[-1]) if len(frame) else 1.0,
    }


def _build_drawdown_diagnostics(performance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    work = performance.copy()
    work["date"] = pd.to_datetime(work["date"])
    for model_name, group in work.groupby("model_name", sort=True):
        ordered = group.sort_values("date")
        full = _period_metrics(ordered, "full_sample")
        full["model_name"] = model_name
        rows.append(full)
        for year, year_group in ordered.groupby(ordered["date"].dt.year):
            metrics = _period_metrics(year_group, str(year))
            metrics["model_name"] = model_name
            rows.append(metrics)
        stress = ordered[
            (ordered["date"] >= pd.Timestamp("2023-07-01"))
            & (ordered["date"] <= pd.Timestamp("2025-12-31"))
        ]
        if not stress.empty:
            metrics = _period_metrics(stress, "2023H2_to_2025")
            metrics["model_name"] = model_name
            rows.append(metrics)
    return pd.DataFrame(rows)


def _summarize_strategy(
    model_name: str,
    positions: pd.DataFrame,
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    performance = compute_portfolio_return(
        positions,
        returns,
        commission_bps=COMMISSION_BPS,
        slippage_bps=SLIPPAGE_BPS,
    )
    performance = apply_volatility_target(
        performance, target_vol=0.25, lookback=20, max_leverage=1.0
    )
    performance.insert(0, "model_name", model_name)

    raw_summary = summarize_backtest(
        performance.rename(columns={"portfolio_return": "portfolio_return"}), positions
    )
    raw_summary.insert(0, "model_name", f"{model_name}_raw")

    scaled_performance = performance[["date", "scaled_portfolio_return"]].rename(
        columns={"scaled_portfolio_return": "portfolio_return"}
    )
    scaled_performance["cum_nav"] = (1.0 + scaled_performance["portfolio_return"]).cumprod()
    scaled_performance["drawdown"] = (
        scaled_performance["cum_nav"] / scaled_performance["cum_nav"].cummax() - 1.0
    )
    scaled_summary = summarize_backtest(scaled_performance, positions)
    scaled_summary.insert(0, "model_name", f"{model_name}_vol_target")
    return performance, pd.concat([raw_summary, scaled_summary], ignore_index=True)


if __name__ == "__main__":
    cfg = ProjectConfig.from_root(Path(__file__).resolve().parents[1])
    panel = pd.read_parquet(cfg.processed_root / "research_panel.parquet")
    returns = panel[["date", "security", "fwd_ret_1d"]].drop_duplicates()
    selected = pd.read_csv(cfg.output_root / "enhanced_selection" / "selected_factors.csv")

    exposures, factor_names = _prepare_selected_exposures(panel, selected, cfg.factor_root)
    premia = estimate_fama_macbeth_premia(exposures, returns, factor_names, min_obs=100)
    weights = build_rolling_factor_weights(
        premia, factor_names, lookback=252, min_periods=60, max_weight=0.35
    )
    score = apply_factor_weights(exposures, weights, factor_names)
    risk_frame = _build_risk_frame(panel)
    daily_positions = _build_positions(score, top_frac=0.10, rebalance_every=1)
    weekly_positions = _build_positions(score, top_frac=0.10, rebalance_every=5)
    smoothed_positions = smooth_positions(daily_positions, alpha=SMOOTHING_ALPHA)
    optimizer_target_positions = build_optimized_long_short_positions(
        score,
        risk_frame,
        top_frac=0.10,
        max_abs_weight=0.03,
    )
    optimizer_positions = smooth_positions(optimizer_target_positions, alpha=SMOOTHING_ALPHA)

    daily_performance, daily_summary = _summarize_strategy(
        "dynamic_daily", daily_positions, returns
    )
    weekly_performance, weekly_summary = _summarize_strategy(
        "dynamic_weekly", weekly_positions, returns
    )
    smoothed_performance, smoothed_summary = _summarize_strategy(
        f"dynamic_smoothed_alpha{SMOOTHING_ALPHA:.2f}".replace(".", "p"),
        smoothed_positions,
        returns,
    )
    optimizer_performance, optimizer_summary = _summarize_strategy(
        f"dynamic_optimizer_alpha{SMOOTHING_ALPHA:.2f}".replace(".", "p"),
        optimizer_positions,
        returns,
    )
    performance = pd.concat(
        [daily_performance, weekly_performance, smoothed_performance, optimizer_performance],
        ignore_index=True,
    )
    summary = pd.concat(
        [daily_summary, weekly_summary, smoothed_summary, optimizer_summary],
        ignore_index=True,
    )
    diagnostics = _build_drawdown_diagnostics(performance)

    output_dir = cfg.output_root / "enhanced_backtest"
    output_dir.mkdir(parents=True, exist_ok=True)
    performance.to_csv(output_dir / "strategy_daily_returns.csv", index=False)
    summary.to_csv(output_dir / "strategy_summary.csv", index=False)
    daily_positions.to_csv(output_dir / "positions_daily.csv", index=False)
    weekly_positions.to_csv(output_dir / "positions_weekly.csv", index=False)
    smoothed_positions.to_csv(output_dir / "positions_smoothed_alpha65.csv", index=False)
    optimizer_target_positions.to_csv(output_dir / "positions_optimizer_target.csv", index=False)
    optimizer_positions.to_csv(output_dir / "positions_optimizer_alpha65.csv", index=False)
    diagnostics.to_csv(output_dir / "drawdown_diagnostics.csv", index=False)
    premia.to_csv(output_dir / "factor_premia.csv", index=False)
    weights.to_csv(output_dir / "rolling_factor_weights.csv", index=False)
    print(output_dir / "strategy_summary.csv")
