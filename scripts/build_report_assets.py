from pathlib import Path

import pandas as pd

from mfe5210_alpha.config import ProjectConfig
from mfe5210_alpha.report.assets import (
    build_summary_rows,
    write_dataframe_table,
    write_key_value_table,
)
from mfe5210_alpha.report.appendices import (
    write_drawdown_diagnostics,
    write_correlation_matrix,
    write_factor_dictionary,
    write_operator_definitions,
    write_selection_details,
    write_single_factor_full_results,
)
from mfe5210_alpha.report.figures import (
    plot_factor_sharpe_distribution,
    plot_selected_correlation,
    plot_selected_family_counts,
    plot_strategy_drawdown,
    plot_strategy_nav,
)
from mfe5210_alpha.report.references import write_bibtex


PRIMARY_MODEL_CANDIDATES = [
    "dynamic_smoothed_alpha0p65_vol_target",
    "dynamic_optimizer_alpha0p65_vol_target",
    "dynamic_daily_vol_target",
]
PRIMARY_DAILY_MODEL_CANDIDATES = [
    "dynamic_smoothed_alpha0p65",
    "dynamic_optimizer_alpha0p65",
    "dynamic_daily",
]


def _format_selected_factors(selected: pd.DataFrame) -> pd.DataFrame:
    table = selected.copy()
    for column in ["aligned_ls_ann_sharpe", "aligned_rank_ic_ir", "research_score"]:
        table[column] = table[column].map(lambda value: f"{float(value):.3f}")
    table["rev_flag"] = table["rev_flag"].map(lambda value: "Yes" if bool(value) else "No")
    return table


def _format_strategy_comparison(strategy_summary: pd.DataFrame) -> pd.DataFrame:
    table = strategy_summary.copy()
    for column in [
        "annualized_return",
        "annualized_sharpe",
        "max_drawdown",
        "average_one_way_turnover",
    ]:
        table[column] = table[column].map(lambda value: f"{float(value):.3f}")
    return table


def _extract_primary_daily_returns(strategy_daily: pd.DataFrame) -> pd.DataFrame:
    daily = pd.DataFrame()
    for model_name in PRIMARY_DAILY_MODEL_CANDIDATES:
        daily = strategy_daily.loc[strategy_daily["model_name"] == model_name].copy()
        if not daily.empty:
            break
    if daily.empty:
        daily = strategy_daily.copy()
    daily["portfolio_return"] = daily["scaled_portfolio_return"]
    daily["cum_nav"] = daily["scaled_cum_nav"]
    daily["drawdown"] = daily["scaled_drawdown"]
    return daily


def _primary_model_name(strategy_summary: pd.DataFrame) -> str:
    if "model_name" not in strategy_summary.columns:
        return ""
    available = set(strategy_summary["model_name"])
    for model_name in PRIMARY_MODEL_CANDIDATES:
        if model_name in available:
            return model_name
    return str(strategy_summary["model_name"].iloc[0])


if __name__ == "__main__":
    cfg = ProjectConfig.from_root(Path(__file__).resolve().parents[1])
    report_root = cfg.report_root
    figure_root = report_root / "assets" / "figures"
    table_root = report_root / "assets" / "tables"

    single = pd.read_csv(cfg.output_root / "enhanced_single_factor" / "single_factor_summary.csv")
    selected = pd.read_csv(cfg.output_root / "enhanced_selection" / "selected_factors.csv")
    catalog = pd.read_csv(cfg.factor_root / "factor_catalog.csv")
    selection_summary = pd.read_csv(
        cfg.output_root / "enhanced_selection" / "selection_summary.csv"
    )
    selection_details_path = cfg.output_root / "enhanced_selection" / "selection_details.csv"
    selection_details = (
        pd.read_csv(selection_details_path) if selection_details_path.exists() else selected.copy()
    )
    selected_corr = pd.read_csv(
        cfg.output_root / "enhanced_selection" / "selected_factor_correlation_matrix.csv",
        index_col=0,
    )
    strategy_summary = pd.read_csv(
        cfg.output_root / "enhanced_backtest" / "strategy_summary.csv"
    )
    strategy_daily = pd.read_csv(
        cfg.output_root / "enhanced_backtest" / "strategy_daily_returns.csv"
    )
    primary_daily = _extract_primary_daily_returns(strategy_daily)
    primary_model = _primary_model_name(strategy_summary)
    single["aligned_ann_sharpe"] = single["ann_sharpe"] * single["rev_flag"].map(
        lambda flag: -1.0 if bool(flag) else 1.0
    )

    write_key_value_table(
        build_summary_rows(
            selection_summary,
            strategy_summary,
            single,
            primary_model_name=primary_model,
        ),
        table_root / "summary_metrics.tex",
    )
    write_dataframe_table(
        _format_strategy_comparison(strategy_summary),
        [
            "model_name",
            "annualized_return",
            "annualized_sharpe",
            "max_drawdown",
            "average_one_way_turnover",
        ],
        table_root / "strategy_comparison.tex",
        max_rows=10,
    )
    write_dataframe_table(
        _format_selected_factors(selected),
        [
            "factor_name",
            "family",
            "aligned_ls_ann_sharpe",
            "aligned_rank_ic_ir",
            "research_score",
            "rev_flag",
        ],
            table_root / "selected_factors.tex",
        max_rows=20,
    )
    appendix_single = single.merge(
        selection_details[
            [
                column
                for column in [
                    "factor_name",
                    "robust_score",
                    "recent_ls_sharpe",
                    "factor_max_drawdown",
                    "cluster_id",
                    "cluster_rank",
                    "selection_stage",
                ]
                if column in selection_details.columns
            ]
        ],
        on="factor_name",
        how="left",
    )
    write_factor_dictionary(
        catalog,
        table_root / "factor_dictionary.tex",
        factor_stats=single,
    )
    write_operator_definitions(table_root / "operator_definitions.tex")
    write_single_factor_full_results(
        appendix_single,
        table_root / "single_factor_full_results.tex",
    )
    write_selection_details(selection_details, table_root / "selection_details.tex")
    write_correlation_matrix(selected_corr, table_root / "fixed_factor_correlation.tex")
    diagnostics_path = cfg.output_root / "enhanced_backtest" / "drawdown_diagnostics.csv"
    if diagnostics_path.exists():
        write_drawdown_diagnostics(
            pd.read_csv(diagnostics_path),
            table_root / "drawdown_diagnostics.tex",
        )

    plot_strategy_nav(primary_daily, figure_root / "strategy_nav.png")
    plot_strategy_drawdown(primary_daily, figure_root / "strategy_drawdown.png")
    plot_factor_sharpe_distribution(single, figure_root / "single_factor_sharpe_distribution.png")
    plot_selected_correlation(selected_corr, figure_root / "selected_factor_correlation.png")
    plot_selected_family_counts(selected, figure_root / "selected_factor_families.png")
    write_bibtex(report_root / "references.bib")

    print(report_root / "assets")
