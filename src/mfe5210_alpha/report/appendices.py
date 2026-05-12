from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from mfe5210_alpha.report.assets import latex_escape


DATA_FIELDS = [
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "amount",
    "turn",
    "ret_1d",
    "fwd_ret_1d",
    "adv20",
    "peTTM",
    "pbMRQ",
    "psTTM",
    "pcfNcfTTM",
]


def extract_formula_fields(formula: str) -> str:
    hits = []
    for field in DATA_FIELDS:
        match = re.search(rf"(?<![A-Za-z0-9_]){re.escape(field)}(?![A-Za-z0-9_])", formula)
        if match:
            hits.append((match.start(), field))
    return ", ".join(field for _, field in sorted(hits))


def operator_definitions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("delay", "delay(x,n): security-level lag of field x by n trading days."),
            ("mean", "mean(x,n): security-level rolling arithmetic mean over n trading days."),
            ("std", "std(x,n): security-level rolling standard deviation over n trading days."),
            ("diff", "diff(x,n): security-level x_t minus x_{t-n}."),
            ("pct_change", "pct_change(x,n): security-level x_t / x_{t-n} - 1."),
            ("zscore", "zscore(x): cross-sectional daily z-score using same-date mean and population standard deviation."),
            ("winsorize", "winsorize(x): cross-sectional daily clipping at the 1st and 99th percentiles."),
            ("rankIC", "Daily Spearman rank correlation between factor exposure and next-day return."),
            ("IC", "Daily Pearson correlation between factor exposure and next-day return."),
            ("residualize(x,Z)", "Daily OLS residual from regressing factor x on control matrix Z."),
        ],
        columns=["operator", "definition"],
    )


def _write_longtable(
    frame: pd.DataFrame,
    columns: list[str],
    headers: list[str],
    output_path: Path,
    spec: str,
    size: str = "\\scriptsize",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        size,
        "\\setlength{\\tabcolsep}{2pt}",
        f"\\begin{{longtable}}{{{spec}}}",
        "\\toprule",
        " & ".join(latex_escape(header) for header in headers) + " \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        " & ".join(latex_escape(header) for header in headers) + " \\\\",
        "\\midrule",
        "\\endhead",
    ]
    for _, row in frame.loc[:, columns].iterrows():
        lines.append(" & ".join(latex_escape(row[column]) for column in columns) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{longtable}", "\\normalsize", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def write_factor_dictionary(
    catalog: pd.DataFrame,
    output_path: Path,
    factor_stats: pd.DataFrame | None = None,
) -> Path:
    table = catalog.copy()
    if factor_stats is not None:
        aligned = factor_stats[["factor_name", "aligned_ann_sharpe"]].copy()
        aligned["aligned_ann_sharpe"] = aligned["aligned_ann_sharpe"].map(
            lambda value: f"{float(value):.3f}"
        )
        table = table.merge(aligned, on="factor_name", how="left")
    else:
        table["aligned_ann_sharpe"] = ""
    table["fields"] = table["formula_text"].map(extract_formula_fields)
    return _write_longtable(
        table,
        ["factor_name", "family", "aligned_ann_sharpe", "fields", "formula_text"],
        ["Factor", "Family", "Aligned Sharpe", "Fields", "Exact formula"],
        output_path,
        spec="p{0.17\\textwidth}p{0.10\\textwidth}p{0.10\\textwidth}p{0.16\\textwidth}p{0.35\\textwidth}",
    )


def write_operator_definitions(output_path: Path) -> Path:
    return _write_longtable(
        operator_definitions(),
        ["operator", "definition"],
        ["Operator", "Definition"],
        output_path,
        spec="p{0.20\\textwidth}p{0.70\\textwidth}",
    )


def _format_bool(value: object) -> str:
    return "Yes" if bool(value) else "No"


def write_single_factor_full_results(summary: pd.DataFrame, output_path: Path) -> Path:
    table = summary.copy()
    for column in [
        "ann_sharpe",
        "ls_ann_sharpe",
        "ic_mean",
        "rank_ic_mean",
        "ic_ir",
        "rank_ic_ir",
        "monthly_win_rate",
        "robust_score",
    ]:
        if column not in table.columns:
            table[column] = 0.0
        table[column] = table[column].map(lambda value: f"{float(value):.3f}")
    if "rev_flag" not in table.columns:
        table["rev_flag"] = False
    table["rev_flag"] = table["rev_flag"].map(_format_bool)
    return _write_longtable(
        table,
        [
            "factor_name",
            "ann_sharpe",
            "ls_ann_sharpe",
            "ic_mean",
            "rank_ic_mean",
            "ic_ir",
            "rank_ic_ir",
            "monthly_win_rate",
            "robust_score",
            "rev_flag",
        ],
        [
            "Factor",
            "Sharpe",
            "LS Sharpe",
            "IC",
            "RankIC",
            "ICIR",
            "RankICIR",
            "WinRate",
            "Robust",
            "Rev",
        ],
        output_path,
        spec="lrrrrrrrrl",
    )


def write_selection_details(details: pd.DataFrame, output_path: Path) -> Path:
    table = details.copy()
    for column in ["robust_score", "recent_ls_sharpe", "factor_max_drawdown"]:
        if column not in table.columns:
            table[column] = 0.0
        table[column] = table[column].map(lambda value: f"{float(value):.3f}")
    return _write_longtable(
        table,
        [
            "factor_name",
            "cluster_id",
            "cluster_rank",
            "robust_score",
            "recent_ls_sharpe",
            "factor_max_drawdown",
            "selection_stage",
        ],
        ["Factor", "Cluster", "Rank", "Robust", "Recent", "MDD", "Stage"],
        output_path,
        spec="lrrrrrl",
    )


def write_drawdown_diagnostics(diagnostics: pd.DataFrame, output_path: Path) -> Path:
    table = diagnostics.copy()
    for column in ["annualized_return", "annualized_sharpe", "max_drawdown", "ending_nav"]:
        if column not in table.columns:
            table[column] = 0.0
        table[column] = table[column].map(lambda value: f"{float(value):.3f}")
    return _write_longtable(
        table,
        [
            "model_name",
            "period",
            "annualized_return",
            "annualized_sharpe",
            "max_drawdown",
            "ending_nav",
        ],
        ["Model", "Period", "AnnRet", "Sharpe", "MDD", "End NAV"],
        output_path,
        spec="llrrrr",
    )


def write_correlation_matrix(matrix: pd.DataFrame, output_path: Path) -> Path:
    table = matrix.copy().round(3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = "l" + "r" * len(table.columns)
    lines = [
        "\\scriptsize",
        f"\\begin{{tabular}}{{{header}}}",
        "\\toprule",
        "Factor & " + " & ".join(latex_escape(column) for column in table.columns) + " \\\\",
        "\\midrule",
    ]
    for index, row in table.iterrows():
        values = " & ".join(f"{float(value):.3f}" for value in row.tolist())
        lines.append(f"{latex_escape(index)} & {values} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\normalsize", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
