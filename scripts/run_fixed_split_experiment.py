from pathlib import Path

import pandas as pd

from mfe5210_alpha.config import ProjectConfig
from mfe5210_alpha.experiments.fixed_split import run_fixed_split_experiment


def _daily_signal_diagnostics_path(cfg: ProjectConfig) -> Path:
    primary = cfg.output_root / "single_factor" / "daily_signal_diagnostics.parquet"
    if primary.exists():
        return primary
    return cfg.output_root / "enhanced_single_factor" / "daily_signal_diagnostics.parquet"


if __name__ == "__main__":
    cfg = ProjectConfig.from_root(Path(__file__).resolve().parents[1])
    panel = pd.read_parquet(cfg.processed_root / "research_panel.parquet")
    signal_diagnostics = pd.read_parquet(_daily_signal_diagnostics_path(cfg))

    result = run_fixed_split_experiment(panel, signal_diagnostics, cfg.factor_root, cfg)
    output_dir = cfg.output_root / "fixed_split"
    output_dir.mkdir(parents=True, exist_ok=True)
    result["performance"].to_csv(output_dir / "test_daily_returns.csv", index=False)
    result["positions"].to_csv(output_dir / "test_positions.csv", index=False)
    result["summary"].to_csv(output_dir / "strategy_summary.csv", index=False)
    result["selection_details"].to_csv(output_dir / "selection_details.csv", index=False)
    result["weights"].to_csv(output_dir / "fixed_factor_weights.csv", index=False)
    print(output_dir / "test_daily_returns.csv")
