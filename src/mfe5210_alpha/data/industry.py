from __future__ import annotations

import pandas as pd


def normalize_baostock_industry(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    if "industryClassification" in frame.columns:
        frame = frame.rename(columns={"industryClassification": "industry_classification"})
    required = ["code", "code_name", "industry", "industry_classification", "asof_date", "source"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing industry columns: {missing}")

    frame["asof_date"] = pd.to_datetime(frame["asof_date"])
    frame = frame.rename(columns={"code": "security"})
    columns = ["security", "code_name", "industry", "industry_classification", "asof_date", "source"]
    return frame[columns].drop_duplicates(subset=["security"]).sort_values("security").reset_index(drop=True)


def attach_industry_classification(panel: pd.DataFrame, industry: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    static_columns = ["code_name", "industry", "industry_classification", "asof_date", "source"]
    mapping_columns = ["security"] + [column for column in static_columns if column in industry.columns]
    mapping = industry[mapping_columns].copy()
    out = frame.merge(mapping, on="security", how="left")
    out["industry_is_static"] = out["industry"].notna()
    return out
