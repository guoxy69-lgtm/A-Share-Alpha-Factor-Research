from __future__ import annotations

import pandas as pd


def build_composite_score(factor_panel: pd.DataFrame) -> pd.DataFrame:
    score = (
        factor_panel.groupby(["date", "security"])["aligned_factor_value"]
        .mean()
        .reset_index()
    )
    return score.rename(columns={"aligned_factor_value": "score"})


def build_equal_weight_score(exposures: pd.DataFrame, factor_names: list[str]) -> pd.DataFrame:
    columns = ["date", "security", "score"]
    if exposures.empty or not factor_names:
        return pd.DataFrame(columns=columns)

    score = exposures[["date", "security"]].copy()
    score["score"] = exposures[factor_names].mean(axis=1, skipna=True)
    return score.dropna(subset=["score"])[columns]


def build_weighted_score(
    exposures: pd.DataFrame,
    weights: pd.DataFrame,
    factor_names: list[str],
) -> pd.DataFrame:
    columns = ["date", "security", "score"]
    if exposures.empty or not factor_names:
        return pd.DataFrame(columns=columns)

    merged = exposures.merge(weights, on="date", how="left", suffixes=("", "_weight"))
    score = pd.Series(0.0, index=merged.index)
    fallback = 1.0 / len(factor_names)
    for name in factor_names:
        score = score + merged[name] * merged[f"{name}_weight"].fillna(fallback)
    return merged[["date", "security"]].assign(score=score).dropna(subset=["score"])[columns]
