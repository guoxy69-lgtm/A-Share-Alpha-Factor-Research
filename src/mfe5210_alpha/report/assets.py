from __future__ import annotations

from pathlib import Path

import pandas as pd


LATEX_ESCAPE_MAP = {
    "&": "\\&",
    "%": "\\%",
    "$": "\\$",
    "#": "\\#",
    "_": "\\_",
    "{": "\\{",
    "}": "\\}",
}


def latex_escape(value: object) -> str:
    text = str(value)
    for raw, escaped in LATEX_ESCAPE_MAP.items():
        text = text.replace(raw, escaped)
    return text


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def build_summary_rows(
    selection_summary: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    single_factor_summary: pd.DataFrame,
    primary_model_name: str = None,
) -> list[tuple[str, str]]:
    selection = selection_summary.iloc[0]
    if primary_model_name and "model_name" in strategy_summary.columns:
        match = strategy_summary.loc[strategy_summary["model_name"] == primary_model_name]
        strategy = match.iloc[0] if not match.empty else strategy_summary.iloc[0]
    else:
        strategy = strategy_summary.iloc[0]
    return [
        ("Candidate factor count", str(len(single_factor_summary))),
        ("Selected factor count", str(int(selection["selected_factor_count"]))),
        ("Maximum absolute factor correlation", _fmt(float(selection["max_abs_corr"]))),
        ("Average single-factor Sharpe", _fmt(float(single_factor_summary["ann_sharpe"].mean()))),
        ("Reported model", strategy.get("model_name", "long-short")),
        ("Long-short annualized return", _fmt(float(strategy["annualized_return"]))),
        ("Long-short annualized Sharpe", _fmt(float(strategy["annualized_sharpe"]))),
        ("Maximum drawdown", _fmt(float(strategy["max_drawdown"]))),
        ("Average one-way turnover", _fmt(float(strategy["average_one_way_turnover"]))),
    ]


def write_key_value_table(rows: list[tuple[str, str]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{lr}",
        "\\toprule",
        "Metric & Value \\\\",
        "\\midrule",
    ]
    lines.extend(f"{latex_escape(metric)} & {latex_escape(value)} \\\\" for metric, value in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def write_dataframe_table(
    frame: pd.DataFrame,
    columns: list[str],
    output_path: Path,
    max_rows: int = 15,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = frame.loc[:, columns].head(max_rows).copy()
    lines = [
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\toprule",
        " & ".join(latex_escape(column) for column in columns) + " \\\\",
        "\\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(" & ".join(latex_escape(row[column]) for column in columns) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
