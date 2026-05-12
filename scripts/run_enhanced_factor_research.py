from pathlib import Path

import pandas as pd

from mfe5210_alpha.config import ProjectConfig
from mfe5210_alpha.research.neutralization import build_style_controls, neutralize_factor_matrix
from mfe5210_alpha.research.single_factor import (
    compute_daily_factor_series,
    evaluate_factor,
)


CONTROL_COLUMNS = ["size_proxy", "beta_proxy", "volatility_proxy"]
BATCH_SIZE = 10


def batches(items: list[Path], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


if __name__ == "__main__":
    cfg = ProjectConfig.from_root(Path(__file__).resolve().parents[1])
    panel = pd.read_parquet(cfg.processed_root / "research_panel.parquet")
    controls = build_style_controls(panel)
    returns = panel[["date", "security", "fwd_ret_1d"]].drop_duplicates()
    base = returns.merge(controls, on=["date", "security"], how="left")

    summary_rows = []
    factor_daily_frames = []
    ls_daily_frames = []
    factor_paths = [
        path
        for path in sorted(cfg.factor_root.glob("*.parquet"))
        if not path.name.startswith("._")
    ]

    for batch_no, batch_paths in enumerate(batches(factor_paths, BATCH_SIZE), start=1):
        factor_names = [path.stem for path in batch_paths]
        print(f"Neutralizing batch {batch_no}: {', '.join(factor_names)}", flush=True)

        batch_frame = base.copy()
        for path, factor_name in zip(batch_paths, factor_names):
            factor = pd.read_parquet(path)[["date", "security", factor_name]]
            batch_frame = batch_frame.merge(factor, on=["date", "security"], how="left")

        neutral = neutralize_factor_matrix(
            batch_frame,
            factor_cols=factor_names,
            control_cols=CONTROL_COLUMNS,
        )

        for factor_name in factor_names:
            neutral_col = f"{factor_name}_neutral"
            research_frame = neutral[["date", "security", neutral_col, "fwd_ret_1d"]].rename(
                columns={neutral_col: "factor_value"}
            )

            metrics = evaluate_factor(research_frame, factor_name)
            factor_daily, ls_daily = compute_daily_factor_series(
                research_frame, factor_name, bool(metrics["rev_flag"])
            )
            summary_rows.append(metrics)
            factor_daily_frames.append(factor_daily)
            ls_daily_frames.append(ls_daily)

        del batch_frame, neutral

    output_dir = cfg.output_root / "enhanced_single_factor"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(summary_rows).sort_values("ann_sharpe", ascending=False)
    summary.to_csv(output_dir / "single_factor_summary.csv", index=False)
    pd.concat(factor_daily_frames, ignore_index=True).to_parquet(
        output_dir / "daily_factor_returns.parquet", index=False
    )
    pd.concat(ls_daily_frames, ignore_index=True).to_parquet(
        output_dir / "daily_long_short_returns.parquet", index=False
    )
    pd.DataFrame(
        [
            {
                "control_columns": ",".join(CONTROL_COLUMNS),
                "batch_size": BATCH_SIZE,
            }
        ]
    ).to_csv(output_dir / "neutralization_config.csv", index=False)
    print(output_dir / "single_factor_summary.csv")
