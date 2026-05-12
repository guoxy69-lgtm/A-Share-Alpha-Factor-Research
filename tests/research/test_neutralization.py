import numpy as np
import pandas as pd

from mfe5210_alpha.research.neutralization import (
    build_style_controls,
    neutralize_factor,
    neutralize_factor_matrix,
)


def test_build_style_controls_creates_tradable_style_proxies():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03"] * 3),
            "security": ["A", "A", "B", "B", "C", "C"],
            "close": [10.0, 10.5, 20.0, 19.8, 30.0, 30.6],
            "amount": [1000.0, 1100.0, 2000.0, 1900.0, 3000.0, 3200.0],
            "turn": [1.0, 1.1, 2.0, 1.9, 3.0, 3.2],
            "ret_1d": [0.01, 0.05, -0.02, -0.01, 0.03, 0.02],
            "pbMRQ": [1.2, 1.3, 2.0, 2.1, 3.0, 3.2],
        }
    )

    controls = build_style_controls(frame)

    assert {"size_proxy", "liquidity_proxy", "volatility_proxy", "beta_proxy", "valuation_proxy"} <= set(
        controls.columns
    )
    assert controls["size_proxy"].notna().all()
    assert controls["liquidity_proxy"].notna().all()


def test_neutralize_factor_removes_cross_sectional_linear_exposure():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"] * 5 + ["2020-01-03"] * 5),
            "security": list("ABCDE") * 2,
            "factor_value": [2.0, 4.1, 6.0, 8.1, 10.0, 1.0, 3.2, 5.1, 7.2, 9.1],
            "size_proxy": [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    neutralized = neutralize_factor(frame, ["size_proxy"], min_obs=4)

    corr = neutralized.groupby("date")[["neutral_factor_value", "size_proxy"]].corr()
    exposure_corr = corr.loc[(slice(None), "neutral_factor_value"), "size_proxy"].abs()
    assert (exposure_corr < 1e-10).all()


def test_neutralize_factor_preserves_extra_research_columns():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"] * 4),
            "security": list("ABCD"),
            "factor_value": [1.0, 2.0, 3.0, 4.0],
            "size_proxy": [1.0, 2.0, 3.0, 4.0],
            "fwd_ret_1d": [0.01, 0.02, -0.01, -0.02],
        }
    )

    neutralized = neutralize_factor(frame, ["size_proxy"], min_obs=4)

    assert "fwd_ret_1d" in neutralized.columns
    assert neutralized["fwd_ret_1d"].tolist() == frame["fwd_ret_1d"].tolist()


def test_neutralize_factor_matrix_residualizes_multiple_factors_at_once():
    dates = pd.to_datetime(["2020-01-02"] * 8 + ["2020-01-03"] * 8)
    size = np.tile(np.linspace(-1.5, 1.5, 8), 2)
    beta = np.tile(np.linspace(1.5, -1.5, 8), 2)
    residual_shape = np.tile([0.2, -0.1, 0.1, -0.2, 0.15, -0.05, 0.05, -0.15], 2)
    frame = pd.DataFrame(
        {
            "date": dates,
            "security": list("ABCDEFGH") * 2,
            "size_proxy": size,
            "beta_proxy": beta,
            "quality": 2.0 * size - 0.5 * beta + residual_shape,
            "reversal": -1.5 * size + 0.3 * beta - residual_shape,
        }
    )

    neutralized = neutralize_factor_matrix(
        frame,
        factor_cols=["quality", "reversal"],
        control_cols=["size_proxy", "beta_proxy"],
        min_obs=6,
    )

    for neutral_col in ["quality_neutral", "reversal_neutral"]:
        corr = neutralized.groupby("date")[[neutral_col, "size_proxy", "beta_proxy"]].corr()
        assert (
            corr.loc[(slice(None), neutral_col), ["size_proxy", "beta_proxy"]].abs().to_numpy()
            < 1e-10
        ).all()
