from __future__ import annotations

from mfe5210_alpha.factors.base import FactorDefinition


def build_factor_catalog() -> list[FactorDefinition]:
    catalog: list[FactorDefinition] = []

    for window in [5, 10, 20, 60, 120, 240]:
        catalog.extend(
            [
                FactorDefinition(
                    name=f"return_{window}d",
                    family="momentum",
                    description=f"{window}-day cumulative return",
                    formula_text=f"close / delay(close, {window}) - 1",
                    source_quote="self-derived daily momentum factor",
                ),
                FactorDefinition(
                    name=f"reversal_{window}d",
                    family="momentum",
                    description=f"negative {window}-day cumulative return",
                    formula_text=f"-(close / delay(close, {window}) - 1)",
                    source_quote="self-derived daily reversal factor",
                ),
                FactorDefinition(
                    name=f"gap_drift_{window}d",
                    family="momentum",
                    description=f"{window}-day average open-close drift",
                    formula_text=f"mean((close-open)/open, {window})",
                    source_quote="self-derived daily momentum factor",
                ),
            ]
        )

    for window in [5, 10, 20, 40, 60, 120]:
        catalog.extend(
            [
                FactorDefinition(
                    name=f"volatility_{window}d",
                    family="volatility",
                    description=f"{window}-day rolling return volatility",
                    formula_text=f"std(ret_1d, {window})",
                    source_quote="self-derived daily volatility factor",
                ),
                FactorDefinition(
                    name=f"range_vol_{window}d",
                    family="volatility",
                    description=f"{window}-day average high-low range",
                    formula_text=f"mean((high-low)/close, {window})",
                    source_quote="self-derived daily volatility factor",
                ),
            ]
        )

    for window in [5, 10, 20, 40, 60, 120]:
        catalog.extend(
            [
                FactorDefinition(
                    name=f"amount_mean_{window}d",
                    family="liquidity",
                    description=f"{window}-day average trading amount",
                    formula_text=f"mean(amount, {window})",
                    source_quote="self-derived daily liquidity factor",
                ),
                FactorDefinition(
                    name=f"turnover_mean_{window}d",
                    family="liquidity",
                    description=f"{window}-day average turnover",
                    formula_text=f"mean(turn, {window})",
                    source_quote="self-derived daily liquidity factor",
                ),
            ]
        )

    for field in ["peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]:
        catalog.extend(
            [
                FactorDefinition(
                    name=f"{field}_level",
                    family="valuation",
                    description=f"cross-sectional level of {field}",
                    formula_text=field,
                    source_quote="self-derived valuation factor",
                ),
                FactorDefinition(
                    name=f"{field}_zscore",
                    family="valuation",
                    description=f"cross-sectional z-score of {field}",
                    formula_text=f"zscore({field})",
                    source_quote="self-derived valuation factor",
                ),
            ]
        )

    wq_formulas = {
        1: "mean(pct_change(close, 1), 5)",
        2: "((high + low) / 2 - close) / close",
        3: "((close - open) / open) * turn",
        4: "pct_change(amount, 5)",
        5: "(open - mean(vwap_proxy, 10)) * abs(close - vwap_proxy), where vwap_proxy = (high + low + close) / 3",
        6: "diff(turn, 1)",
        7: "((close - preclose) / preclose) / turn",
        8: "pct_change(close, 10) - pct_change(close, 3)",
        9: "((high - close) - (close - low)) / (high - low)",
        10: "mean(amount, 20) / amount",
    }
    for idx in range(1, 11):
        catalog.append(
            FactorDefinition(
                name=f"wq_alpha_{idx:03d}",
                family="wq_like",
                description=f"reproducible daily WQ-style alpha {idx:03d}",
                formula_text=wq_formulas[idx],
                source_quote="worldquant 101 alphas",
            )
        )

    assert len(catalog) >= 60
    return catalog
