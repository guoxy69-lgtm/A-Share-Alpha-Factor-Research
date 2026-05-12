from pathlib import Path

import pandas as pd

from mfe5210_alpha.config import ProjectConfig
from mfe5210_alpha.research.correlation import (
    build_factor_correlation,
    compute_research_score,
)
from mfe5210_alpha.research.robust_selection import select_production_factors

CLUSTER_ABS_CORR = 0.85


if __name__ == "__main__":
    cfg = ProjectConfig.from_root(Path(__file__).resolve().parents[1])
    summary = pd.read_csv(cfg.output_root / "enhanced_single_factor" / "single_factor_summary.csv")
    daily = pd.read_parquet(cfg.output_root / "enhanced_single_factor" / "daily_factor_returns.parquet")
    ls_daily = pd.read_parquet(
        cfg.output_root / "enhanced_single_factor" / "daily_long_short_returns.parquet"
    )
    catalog = pd.read_csv(cfg.factor_root / "factor_catalog.csv")

    summary = compute_research_score(summary)
    corr = build_factor_correlation(daily)
    selected, details = select_production_factors(
        summary,
        corr,
        ls_daily=ls_daily,
        max_abs_corr=0.5,
        cluster_abs_corr=CLUSTER_ABS_CORR,
    )

    out = details.set_index("factor_name").loc[selected].reset_index()
    out = out.merge(catalog, on="factor_name", how="left")
    details = details.merge(catalog, on="factor_name", how="left")
    output_dir = cfg.output_root / "enhanced_selection"
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / "selected_factors.csv", index=False)
    details.to_csv(output_dir / "selection_details.csv", index=False)
    corr.to_csv(output_dir / "factor_correlation_matrix.csv")

    selected_corr = corr.loc[selected, selected]
    selected_corr.to_csv(output_dir / "selected_factor_correlation_matrix.csv")
    off_diag = selected_corr.where(~selected_corr.eq(1.0))
    max_abs_corr = off_diag.abs().max().max()
    max_abs_corr = float(0.0 if pd.isna(max_abs_corr) else max_abs_corr)

    pd.DataFrame(
        [
            {
                "selected_factor_count": len(selected),
                "candidate_factor_count": len(summary),
                "cluster_count": int(details["cluster_id"].nunique()),
                "max_abs_corr": max_abs_corr,
                "cluster_abs_corr": CLUSTER_ABS_CORR,
                "selection_method": "cluster_aware_robust_score_with_final_corr_guard",
            }
        ]
    ).to_csv(output_dir / "selection_summary.csv", index=False)
    print(output_dir / "selected_factors.csv")
