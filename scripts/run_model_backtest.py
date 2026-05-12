from pathlib import Path

import pandas as pd

from mfe5210_alpha.config import ProjectConfig
from mfe5210_alpha.model.backtest import (
    build_long_short_positions,
    compute_portfolio_return,
    summarize_backtest,
)
from mfe5210_alpha.model.composite import build_composite_score


COMMISSION_BPS = 3.6
SLIPPAGE_BPS = 0.0


if __name__ == "__main__":
    cfg = ProjectConfig.from_root(Path(__file__).resolve().parents[1])
    panel = pd.read_parquet(cfg.processed_root / "research_panel.parquet")
    selected = pd.read_csv(cfg.output_root / "selection" / "selected_factors.csv")

    factor_frames = []
    for name in selected["factor_name"]:
        factor = pd.read_parquet(cfg.factor_root / f"{name}.parquet").rename(
            columns={name: "factor_value"}
        )
        rev_flag = bool(selected.loc[selected["factor_name"] == name, "rev_flag"].iloc[0])
        factor["aligned_factor_value"] = factor["factor_value"] * (-1.0 if rev_flag else 1.0)
        factor["factor_name"] = name
        factor_frames.append(factor[["date", "security", "factor_name", "aligned_factor_value"]])

    factor_panel = pd.concat(factor_frames, ignore_index=True)
    score = build_composite_score(factor_panel.dropna(subset=["aligned_factor_value"]))
    position_frames = []
    for date, group in score.groupby("date"):
        pos = build_long_short_positions(group[["security", "score"]], top_frac=0.1)
        pos.insert(0, "date", date)
        position_frames.append(pos)
    positions = pd.concat(position_frames, ignore_index=True)

    returns = panel[["date", "security", "fwd_ret_1d"]].drop_duplicates()
    performance = compute_portfolio_return(
        positions,
        returns,
        commission_bps=COMMISSION_BPS,
        slippage_bps=SLIPPAGE_BPS,
    )
    summary = summarize_backtest(performance, positions)

    output_dir = cfg.output_root / "backtest"
    output_dir.mkdir(parents=True, exist_ok=True)
    performance.to_csv(output_dir / "strategy_daily_returns.csv", index=False)
    summary.to_csv(output_dir / "strategy_summary.csv", index=False)
    print(output_dir / "strategy_daily_returns.csv")
