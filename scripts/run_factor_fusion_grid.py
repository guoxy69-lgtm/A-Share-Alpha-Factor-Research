import argparse
from pathlib import Path

import pandas as pd

from mfe5210_alpha.config import ProjectConfig
from mfe5210_alpha.data.industry import attach_industry_classification
from mfe5210_alpha.experiments.fusion_grid import (
    ExperimentSpec,
    build_method_specs,
    run_fixed_fusion_spec,
    run_walk_forward_fusion_specs,
)


def _daily_signal_diagnostics_path(cfg: ProjectConfig) -> Path:
    primary = cfg.output_root / "single_factor" / "daily_signal_diagnostics.parquet"
    if primary.exists():
        return primary
    return cfg.output_root / "enhanced_single_factor" / "daily_signal_diagnostics.parquet"


def _write_route_outputs(
    output_dir: Path,
    route: str,
    results: list[dict[str, pd.DataFrame]],
) -> None:
    pd.concat([result["summary"] for result in results], ignore_index=True).to_csv(
        output_dir / f"{route}_method_comparison.csv", index=False
    )
    pd.concat([result["performance"] for result in results], ignore_index=True).to_csv(
        output_dir / f"{route}_daily_returns.csv", index=False
    )
    pd.concat([result["selection_details"] for result in results], ignore_index=True).to_csv(
        output_dir / f"{route}_selection_details.csv", index=False
    )
    pd.concat([result["weights"] for result in results], ignore_index=True).to_csv(
        output_dir / f"{route}_factor_weights.csv", index=False
    )
    pd.concat([result["quantiles"] for result in results], ignore_index=True).to_csv(
        output_dir / f"{route}_quantile_returns.csv", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", choices=["specs", "fixed", "rolling", "all"], default="specs")
    args = parser.parse_args()

    cfg = ProjectConfig.from_root(Path(__file__).resolve().parents[1])
    specs = build_method_specs()
    output_dir = cfg.output_root / "fusion_grid"
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = output_dir / "experiment_specs.csv"
    pd.DataFrame([spec.__dict__ for spec in specs]).to_csv(spec_path, index=False)
    print(spec_path)
    if args.route == "specs":
        raise SystemExit(0)

    panel = pd.read_parquet(cfg.processed_root / "research_panel.parquet")
    industry_path = cfg.processed_root / "industry_classification.parquet"
    if industry_path.exists():
        industry = pd.read_parquet(industry_path)
        panel = attach_industry_classification(panel, industry)
    signal_diagnostics = pd.read_parquet(_daily_signal_diagnostics_path(cfg))

    if args.route in {"fixed", "all"}:
        fixed_specs = [spec for spec in specs if spec.route == "fixed"]
        fixed_results = [
            run_fixed_fusion_spec(panel, signal_diagnostics, cfg.factor_root, cfg, spec)
            for spec in fixed_specs
        ]
        _write_route_outputs(output_dir, "fixed", fixed_results)
    if args.route in {"rolling", "all"}:
        rolling_specs = [
            ExperimentSpec(
                name=spec.name.replace("fixed_", "rolling_", 1),
                route="rolling",
                fusion_method=spec.fusion_method,
                top_frac=spec.top_frac,
                exit_buffer_frac=spec.exit_buffer_frac,
                neutralize=spec.neutralize,
                neutralization_method=spec.neutralization_method,
                use_industry=spec.use_industry,
                portfolio_type=spec.portfolio_type,
            )
            for spec in specs
            if spec.route == "fixed"
        ]
        rolling_results = run_walk_forward_fusion_specs(
            panel, signal_diagnostics, cfg.factor_root, cfg, rolling_specs
        )
        _write_route_outputs(output_dir, "rolling", rolling_results)
