import pandas as pd

from mfe5210_alpha.model.dynamic import (
    apply_volatility_target,
    build_rolling_factor_weights,
    estimate_fama_macbeth_premia,
)


def test_estimate_fama_macbeth_premia_recovers_cross_sectional_coefficients():
    exposures = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"] * 5 + ["2020-01-03"] * 5),
            "security": list("ABCDE") * 2,
            "f1": [-2.0, -1.0, 0.0, 1.0, 2.0] * 2,
            "f2": [2.0, 1.0, 0.0, -1.0, -2.0] * 2,
        }
    )
    returns = exposures[["date", "security"]].copy()
    returns["fwd_ret_1d"] = 0.03 * exposures["f1"] - 0.01 * exposures["f2"]

    premia = estimate_fama_macbeth_premia(exposures, returns, ["f1", "f2"], min_obs=5)

    assert list(premia.columns) == ["date", "f1", "f2"]
    assert round(float(premia["f1"].iloc[0]), 6) == 0.02
    assert round(float(premia["f2"].iloc[0]), 6) == -0.02


def test_build_rolling_factor_weights_uses_only_past_premia():
    premia = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=6, freq="D"),
            "f1": [0.10, 0.10, 0.10, 0.10, 0.10, -9.0],
            "f2": [-0.02, -0.02, -0.02, -0.02, -0.02, 9.0],
        }
    )

    weights = build_rolling_factor_weights(
        premia, ["f1", "f2"], lookback=3, min_periods=2, max_weight=0.8
    )

    first = weights.loc[weights["date"] == pd.Timestamp("2020-01-01"), ["f1", "f2"]].iloc[0]
    last = weights.loc[weights["date"] == pd.Timestamp("2020-01-06"), ["f1", "f2"]].iloc[0]
    assert first.to_dict() == {"f1": 0.5, "f2": 0.5}
    assert last["f1"] > last["f2"]


def test_apply_volatility_target_reduces_exposure_after_lookback_without_leverage_up():
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=8, freq="D"),
            "portfolio_return": [0.01, -0.01, 0.02, -0.02, 0.03, -0.03, 0.04, -0.04],
        }
    )

    scaled = apply_volatility_target(daily, target_vol=0.10, lookback=3, max_leverage=1.0)

    assert "scaled_portfolio_return" in scaled.columns
    assert scaled["leverage"].max() <= 1.0
    assert abs(scaled["scaled_portfolio_return"].iloc[-1]) < abs(daily["portfolio_return"].iloc[-1])
