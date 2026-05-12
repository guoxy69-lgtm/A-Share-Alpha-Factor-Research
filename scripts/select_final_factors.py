from pathlib import Path

import pandas as pd

from mfe5210_alpha.config import ProjectConfig
from mfe5210_alpha.research.correlation import (
    build_factor_correlation,
    compute_research_score,
    greedy_select_factors,
)


if __name__ == "__main__":
    cfg = ProjectConfig.from_root(Path(__file__).resolve().parents[1])
    summary = pd.read_csv(cfg.output_root / "single_factor" / "single_factor_summary.csv")
    daily = pd.read_parquet(cfg.output_root / "single_factor" / "daily_factor_returns.parquet")
    catalog = pd.read_csv(cfg.factor_root / "factor_catalog.csv")

    summary = compute_research_score(summary)
    corr = build_factor_correlation(daily)
    selected = greedy_select_factors(summary, corr, max_abs_corr=0.5)

    out = summary[summary["factor_name"].isin(selected)].copy()
    out = out.merge(catalog, on="factor_name", how="left")
    output_dir = cfg.output_root / "selection"
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / "selected_factors.csv", index=False)
    corr.to_csv(output_dir / "factor_correlation_matrix.csv")

    selected_names = out["factor_name"].tolist()
    selected_corr = corr.loc[selected_names, selected_names]
    selected_corr.to_csv(output_dir / "selected_factor_correlation_matrix.csv")

    off_diag = selected_corr.where(~selected_corr.eq(1.0))
    max_abs_corr = off_diag.abs().max().max()
    max_abs_corr = float(0.0 if pd.isna(max_abs_corr) else max_abs_corr)

    pd.DataFrame(
        [{"selected_factor_count": len(selected_names), "max_abs_corr": max_abs_corr}]
    ).to_csv(output_dir / "selection_summary.csv", index=False)
    print(output_dir / "selected_factors.csv")
