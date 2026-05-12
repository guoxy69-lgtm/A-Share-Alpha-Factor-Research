from pathlib import Path

import pandas as pd

from mfe5210_alpha.config import ProjectConfig
from mfe5210_alpha.research.single_factor import (
    compute_daily_factor_series,
    evaluate_factor,
)


if __name__ == "__main__":
    cfg = ProjectConfig.from_root(Path(__file__).resolve().parents[1])
    panel = pd.read_parquet(cfg.processed_root / "research_panel.parquet")

    summary_rows = []
    factor_daily_frames = []
    ls_daily_frames = []

    factor_paths = [
        path
        for path in sorted(cfg.factor_root.glob("*.parquet"))
        if not path.name.startswith("._")
    ]
    for path in factor_paths:
        factor_name = path.stem
        factor = pd.read_parquet(path).rename(columns={factor_name: "factor_value"})
        merged = panel.merge(factor, on=["date", "security"], how="inner")
        merged = merged[["date", "security", "factor_value", "fwd_ret_1d"]]
        metrics = evaluate_factor(merged, factor_name)
        factor_daily, ls_daily = compute_daily_factor_series(
            merged, factor_name, bool(metrics["rev_flag"])
        )
        summary_rows.append(metrics)
        factor_daily_frames.append(factor_daily)
        ls_daily_frames.append(ls_daily)

    output_dir = cfg.output_root / "single_factor"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(summary_rows).sort_values("ann_sharpe", ascending=False)
    summary.to_csv(output_dir / "single_factor_summary.csv", index=False)
    pd.concat(factor_daily_frames, ignore_index=True).to_parquet(
        output_dir / "daily_factor_returns.parquet", index=False
    )
    pd.concat(ls_daily_frames, ignore_index=True).to_parquet(
        output_dir / "daily_long_short_returns.parquet", index=False
    )
    print(output_dir / "single_factor_summary.csv")
