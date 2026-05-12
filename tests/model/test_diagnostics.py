import pandas as pd

from mfe5210_alpha.model.backtest import build_long_only_positions
from mfe5210_alpha.model.diagnostics import compute_quantile_returns


def test_build_long_only_positions_selects_top_bucket():
    scores = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-04"] * 5),
            "security": list("ABCDE"),
            "score": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )

    positions = build_long_only_positions(scores, top_frac=0.4)

    assert set(positions["security"]) == {"A", "B"}
    assert positions["weight"].sum() == 1.0


def test_compute_quantile_returns_reports_q5_minus_q1():
    scores = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-04"] * 10),
            "security": [f"S{i}" for i in range(10)],
            "score": list(range(10)),
        }
    )
    returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-04"] * 10),
            "security": [f"S{i}" for i in range(10)],
            "fwd_ret_1d": [i / 100 for i in range(10)],
        }
    )

    out = compute_quantile_returns(scores, returns, quantiles=5)

    assert "q5_minus_q1" in out.columns
    assert out["q5_minus_q1"].iloc[0] > 0
