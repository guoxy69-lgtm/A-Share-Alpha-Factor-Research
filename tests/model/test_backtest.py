import pandas as pd

from mfe5210_alpha.model.backtest import (
    build_long_short_positions,
    build_rebalanced_long_short_positions,
    compute_portfolio_return,
    smooth_positions,
    summarize_backtest,
)


def test_build_long_short_positions_is_market_neutral():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"] * 4),
            "security": ["A", "B", "C", "D"],
            "score": [3.0, 2.0, -2.0, -3.0],
        }
    )

    positions = build_long_short_positions(frame, top_frac=0.25)

    assert round(positions["weight"].sum(), 10) == 0.0
    assert set(positions["security"]) == {"A", "D"}


def test_build_rebalanced_positions_hold_between_rebalance_dates():
    score = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"] * 4 + ["2020-01-03"] * 4 + ["2020-01-06"] * 4),
            "security": ["A", "B", "C", "D"] * 3,
            "score": [3.0, 2.0, -2.0, -3.0, -3.0, 3.0, 2.0, -2.0, -3.0, 3.0, 2.0, -2.0],
        }
    )

    positions = build_rebalanced_long_short_positions(score, top_frac=0.25, rebalance_every=2)
    day1 = positions.loc[positions["date"] == pd.Timestamp("2020-01-02")]
    day2 = positions.loc[positions["date"] == pd.Timestamp("2020-01-03")]
    day3 = positions.loc[positions["date"] == pd.Timestamp("2020-01-06")]

    assert day1[["security", "weight"]].reset_index(drop=True).equals(
        day2[["security", "weight"]].reset_index(drop=True)
    )
    assert set(day3["security"]) == {"B", "A"}


def test_smooth_positions_blends_target_positions_and_keeps_market_neutrality():
    target = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"] * 2 + ["2020-01-03"] * 2),
            "security": ["A", "D", "B", "C"],
            "weight": [1.0, -1.0, 1.0, -1.0],
        }
    )

    smoothed = smooth_positions(target, alpha=0.5)
    day2 = smoothed.loc[smoothed["date"] == pd.Timestamp("2020-01-03")]

    assert round(day2["weight"].sum(), 10) == 0.0
    assert day2.set_index("security")["weight"].round(6).to_dict() == {
        "A": 0.5,
        "B": 0.5,
        "C": -0.5,
        "D": -0.5,
    }


def test_compute_portfolio_return_tracks_turnover_commission_and_net_return():
    positions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"]),
            "security": ["A", "B", "A", "B"],
            "weight": [1.0, -1.0, 0.5, -0.5],
        }
    )
    returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"]),
            "security": ["A", "B", "A", "B"],
            "fwd_ret_1d": [0.01, -0.01, 0.02, -0.02],
        }
    )

    performance = compute_portfolio_return(positions, returns, commission_bps=3.6, slippage_bps=0.0)

    assert performance["one_way_turnover"].round(6).tolist() == [1.0, 0.5]
    assert performance["commission"].round(6).tolist() == [0.00036, 0.00018]
    assert performance["holding_pnl"].round(6).tolist() == [0.02, 0.02]
    assert performance["trading_pnl"].round(6).tolist() == [0.0, 0.0]
    assert performance["net_portfolio_return"].round(6).tolist() == [0.01964, 0.01982]


def test_summarize_backtest_uses_net_portfolio_return_and_reports_turnover():
    positions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"]),
            "security": ["A", "B", "A", "B"],
            "weight": [1.0, -1.0, 0.5, -0.5],
        }
    )
    performance = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "portfolio_return": [0.01964, 0.01982],
            "net_portfolio_return": [0.01964, 0.01982],
            "drawdown": [0.0, 0.0],
            "one_way_turnover": [1.0, 0.5],
        }
    )

    summary = summarize_backtest(performance, positions)

    assert round(float(summary.iloc[0]["average_one_way_turnover"]), 6) == 0.75
    assert float(summary.iloc[0]["annualized_return"]) > 4.7
