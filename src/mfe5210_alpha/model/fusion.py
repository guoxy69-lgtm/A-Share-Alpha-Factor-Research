from __future__ import annotations

import numpy as np
import pandas as pd

from mfe5210_alpha.model.dynamic import _cap_weights


def _normalize(raw: pd.Series, factor_names: list[str], max_weight: float) -> pd.Series:
    clean = (
        raw.reindex(factor_names)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0)
    )
    if clean.sum() <= 0:
        clean = pd.Series(1.0 / len(factor_names), index=factor_names)
    else:
        clean = clean / clean.sum()
    return _cap_weights(clean, max_weight=max_weight)


def _max_ir_weights(
    factor_names: list[str],
    daily_signal_diagnostics: pd.DataFrame,
    max_weight: float,
) -> pd.Series:
    pivot = (
        daily_signal_diagnostics.pivot(
            index="date", columns="factor_name", values="aligned_long_short_return"
        )
        .reindex(columns=factor_names)
        .dropna(how="all")
        .fillna(0.0)
    )
    if pivot.empty:
        return pd.Series(1.0 / len(factor_names), index=factor_names)

    mu = pivot.mean().to_numpy(dtype=float)
    cov = pivot.cov().to_numpy(dtype=float) + np.eye(len(factor_names)) * 1e-6
    raw = np.linalg.pinv(cov) @ mu
    return _normalize(pd.Series(raw, index=factor_names), factor_names, max_weight=max_weight)


def build_factor_weights(
    method: str,
    factor_names: list[str],
    summary: pd.DataFrame,
    daily_signal_diagnostics: pd.DataFrame,
    max_weight: float = 0.50,
) -> pd.Series:
    if not factor_names:
        return pd.Series(dtype=float)

    indexed = summary.set_index("factor_name") if not summary.empty else pd.DataFrame(index=factor_names)
    if method == "equal_weight":
        return pd.Series(1.0 / len(factor_names), index=factor_names)
    if method == "ls_sharpe_weight":
        return _normalize(indexed.get("ls_ann_sharpe", pd.Series(dtype=float)), factor_names, max_weight)
    if method == "rank_icir_weight":
        return _normalize(indexed.get("rank_ic_ir", pd.Series(dtype=float)), factor_names, max_weight)
    if method == "ic_mean_weight":
        return _normalize(indexed.get("ic_mean", pd.Series(dtype=float)), factor_names, max_weight)
    if method == "robust_score_weight":
        return _normalize(indexed.get("robust_score", pd.Series(dtype=float)), factor_names, max_weight)
    if method == "max_ir":
        return _max_ir_weights(factor_names, daily_signal_diagnostics, max_weight)
    raise ValueError(f"unsupported fusion method: {method}")
