from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize_strength(values: pd.Series) -> pd.Series:
    cleaned = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.fillna(cleaned.median() if cleaned.notna().any() else 0.0)
    shifted = cleaned - cleaned.min()
    if shifted.max() <= 1e-12:
        return pd.Series(1.0, index=values.index)
    return shifted + 1e-6


def _cap_and_normalize(raw_weights: pd.Series, max_abs_weight: float, side_sign: float) -> pd.Series:
    weights = raw_weights.clip(lower=0.0)
    if weights.sum() <= 0:
        weights = pd.Series(1.0, index=raw_weights.index)
    weights = weights / weights.sum()
    max_abs_weight = float(max_abs_weight)
    for _ in range(len(weights) + 1):
        over = weights > max_abs_weight
        if not over.any():
            break
        capped_sum = max_abs_weight * over.sum()
        remaining = weights.loc[~over]
        if remaining.empty or capped_sum >= 1.0:
            weights = weights.clip(upper=max_abs_weight)
            break
        weights.loc[over] = max_abs_weight
        weights.loc[~over] = remaining / remaining.sum() * (1.0 - capped_sum)
    total = weights.sum()
    if total > 0:
        weights = weights / total
    return weights * side_sign


def build_optimized_long_short_positions(
    score_frame: pd.DataFrame,
    risk_frame: pd.DataFrame,
    top_frac: float = 0.10,
    max_abs_weight: float = 0.03,
    min_risk: float = 1e-4,
) -> pd.DataFrame:
    merged = score_frame.merge(risk_frame, on=["date", "security"], how="left")
    merged["risk_proxy"] = (
        pd.to_numeric(merged["risk_proxy"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .clip(lower=min_risk)
    )
    merged["risk_proxy"] = merged.groupby("date")["risk_proxy"].transform(
        lambda x: x.fillna(x.median() if x.notna().any() else 1.0)
    )

    position_frames = []
    for date, group in merged.dropna(subset=["score"]).groupby("date", sort=True):
        bucket_size = max(1, int(len(group) * top_frac))
        ordered = group.sort_values("score", ascending=False)
        long_group = ordered.head(bucket_size).copy()
        short_group = ordered.tail(bucket_size).copy()

        long_strength = _normalize_strength(long_group["score"])
        short_strength = _normalize_strength(-short_group["score"])
        long_raw = long_strength / (long_group["risk_proxy"] ** 2)
        short_raw = short_strength / (short_group["risk_proxy"] ** 2)

        long_weights = _cap_and_normalize(long_raw, max_abs_weight, side_sign=1.0)
        short_weights = _cap_and_normalize(short_raw, max_abs_weight, side_sign=-1.0)

        positions = pd.concat(
            [
                pd.DataFrame({"security": long_group["security"].values, "weight": long_weights.values}),
                pd.DataFrame({"security": short_group["security"].values, "weight": short_weights.values}),
            ],
            ignore_index=True,
        )
        positions.insert(0, "date", date)
        position_frames.append(positions)

    if not position_frames:
        return pd.DataFrame(columns=["date", "security", "weight"])
    return pd.concat(position_frames, ignore_index=True)
