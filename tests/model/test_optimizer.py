import pandas as pd

from mfe5210_alpha.model.optimizer import build_optimized_long_short_positions


def test_optimizer_builds_market_neutral_capped_positions():
    score = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"] * 6),
            "security": list("ABCDEF"),
            "score": [3.0, 2.0, 1.0, -1.0, -2.0, -3.0],
        }
    )
    risk = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"] * 6),
            "security": list("ABCDEF"),
            "risk_proxy": [0.2] * 6,
        }
    )

    positions = build_optimized_long_short_positions(
        score,
        risk,
        top_frac=0.34,
        max_abs_weight=0.7,
    )

    assert round(positions["weight"].sum(), 10) == 0.0
    assert round(positions.loc[positions["weight"] > 0, "weight"].sum(), 10) == 1.0
    assert round(positions.loc[positions["weight"] < 0, "weight"].sum(), 10) == -1.0
    assert positions["weight"].abs().max() <= 0.7


def test_optimizer_gives_lower_weight_to_higher_risk_name_with_same_side_signal():
    score = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"] * 6),
            "security": list("ABCDEF"),
            "score": [3.0, 3.0, 1.0, -1.0, -2.0, -3.0],
        }
    )
    risk = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"] * 6),
            "security": list("ABCDEF"),
            "risk_proxy": [0.4, 0.2, 0.2, 0.2, 0.2, 0.2],
        }
    )

    positions = build_optimized_long_short_positions(
        score,
        risk,
        top_frac=0.34,
        max_abs_weight=0.9,
    )
    weights = positions.set_index("security")["weight"]

    assert weights["A"] < weights["B"]
