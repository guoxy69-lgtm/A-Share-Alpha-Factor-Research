from pathlib import Path

import pandas as pd

from mfe5210_alpha.config import ProjectConfig
from mfe5210_alpha.data.sample_diagnostics import summarize_sample_coverage


if __name__ == "__main__":
    cfg = ProjectConfig.from_root(Path(__file__).resolve().parents[1])
    panel = pd.read_parquet(cfg.processed_root / "research_panel.parquet")
    output_path = cfg.output_root / "data_checks" / "sample_coverage.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summarize_sample_coverage(panel).to_csv(output_path, index=False)
    print(output_path)
