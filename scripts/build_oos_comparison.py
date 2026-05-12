from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from mfe5210_alpha.config import ProjectConfig


def summarize_experiment(
    perf: pd.DataFrame,
    summary: pd.DataFrame,
    experiment: str,
) -> dict[str, object]:
    daily = perf["portfolio_return"] if "portfolio_return" in perf.columns else pd.Series(dtype=float)
    final_nav = float((1.0 + daily).prod()) if len(daily) else 1.0
    years = len(daily) / 243 if len(daily) else 0.0
    cagr = final_nav ** (1.0 / years) - 1.0 if years > 0 else 0.0
    annualized_volatility = (
        float(daily.std(ddof=0) * math.sqrt(243)) if len(daily) else 0.0
    )
    row = summary.iloc[0].to_dict() if not summary.empty else {}
    return {
        "experiment": experiment,
        "start_date": perf["date"].min() if "date" in perf.columns and not perf.empty else pd.NaT,
        "end_date": perf["date"].max() if "date" in perf.columns and not perf.empty else pd.NaT,
        "trading_days": len(perf),
        "arithmetic_annualized_return": float(row.get("annualized_return", 0.0) or 0.0),
        "cagr": cagr,
        "annualized_volatility": annualized_volatility,
        "annualized_sharpe": float(row.get("annualized_sharpe", 0.0) or 0.0),
        "max_drawdown": float(row.get("max_drawdown", 0.0) or 0.0),
        "average_one_way_turnover": float(row.get("average_one_way_turnover", 0.0) or 0.0),
        "final_nav": final_nav,
    }


if __name__ == "__main__":
    cfg = ProjectConfig.from_root(Path(__file__).resolve().parents[1])
    fixed_perf = pd.read_csv(
        cfg.output_root / "fixed_split" / "test_daily_returns.csv",
        parse_dates=["date"],
    )
    fixed_summary = pd.read_csv(cfg.output_root / "fixed_split" / "strategy_summary.csv")
    walk_perf = pd.read_csv(
        cfg.output_root / "walk_forward" / "oos_daily_returns.csv",
        parse_dates=["date"],
    )
    walk_summary = pd.read_csv(cfg.output_root / "walk_forward" / "strategy_summary.csv")

    out = pd.DataFrame(
        [
            summarize_experiment(fixed_perf, fixed_summary, "fixed_train_test"),
            summarize_experiment(walk_perf, walk_summary, "rolling_train_test"),
        ]
    )
    output_dir = cfg.output_root / "oos_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "experiment_comparison.csv"
    out.to_csv(output_path, index=False)
    print(output_path)
