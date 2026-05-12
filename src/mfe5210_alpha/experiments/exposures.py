from __future__ import annotations

from pathlib import Path

import pandas as pd

from mfe5210_alpha.research.neutralization import build_style_controls, neutralize_exposures
from mfe5210_alpha.research.preprocess import winsorize_by_date, zscore_by_date


def prepare_weighted_exposures(
    panel: pd.DataFrame,
    factor_root: Path,
    factor_names: list[str],
    factor_signs: dict[str, int],
    neutralize: bool = False,
    neutralization_method: str = "ols",
    use_industry: bool = True,
) -> pd.DataFrame:
    exposures = panel[["date", "security"]].drop_duplicates().copy()
    for factor_name in factor_names:
        factor = pd.read_parquet(factor_root / f"{factor_name}.parquet")[
            ["date", "security", factor_name]
        ].copy()
        factor["date"] = pd.to_datetime(factor["date"])
        factor[factor_name] = factor[factor_name] * int(factor_signs[factor_name])
        temp = factor[["date", factor_name]].rename(columns={factor_name: "factor_value"})
        temp["factor_value"] = winsorize_by_date(temp, "factor_value")
        factor[factor_name] = zscore_by_date(temp, "factor_value")
        exposures = exposures.merge(factor, on=["date", "security"], how="left")
    exposures = exposures.dropna(subset=factor_names, how="all")
    if neutralize and factor_names:
        key_columns = ["date", "security"]
        sort_columns = key_columns + (["industry"] if "industry" in panel.columns else [])
        panel_for_controls = (
            panel.sort_values(sort_columns)
            .drop_duplicates(key_columns, keep="first")
        )
        controls = build_style_controls(panel_for_controls)
        if use_industry and "industry" in panel.columns:
            controls = controls.merge(
                panel_for_controls[key_columns + ["industry"]],
                on=key_columns,
                how="left",
            )
        exposures = neutralize_exposures(
            exposures,
            controls,
            factor_names,
            use_industry=use_industry,
            method=neutralization_method,
        )
    return exposures
