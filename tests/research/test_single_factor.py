import pandas as pd

from mfe5210_alpha.research.single_factor import compute_daily_factor_series, evaluate_factor


def test_evaluate_factor_reports_ic_rankic_and_rev_flag():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-03",
                    "2020-01-03",
                    "2020-01-03",
                ]
            ),
            "security": ["A", "B", "C", "D", "A", "B", "C", "D"],
            "factor_value": [2.0, 1.0, -1.0, -2.0, 2.5, 1.2, -1.1, -2.2],
            "fwd_ret_1d": [0.03, 0.01, -0.01, -0.03, 0.04, 0.015, -0.015, -0.04],
        }
    )

    result = evaluate_factor(frame, factor_name="toy_factor")

    assert result["factor_name"] == "toy_factor"
    assert "ann_sharpe" in result
    assert "ls_ann_sharpe" in result
    assert "ic_mean" in result
    assert "rank_ic_mean" in result
    assert "ic_ir" in result
    assert "rank_ic_ir" in result
    assert "monthly_win_rate" in result
    assert isinstance(result["rev_flag"], bool)
    assert result["ic_mean"] > 0
    assert result["rank_ic_mean"] > 0


def test_daily_series_align_negative_signal_when_rev_flag_true():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-03",
                    "2020-01-03",
                    "2020-01-03",
                ]
            ),
            "security": ["A", "B", "C", "D", "A", "B", "C", "D"],
            "factor_value": [2.0, 1.0, -1.0, -2.0, 2.0, 1.0, -1.0, -2.0],
            "fwd_ret_1d": [-0.03, -0.01, 0.01, 0.03, -0.02, -0.01, 0.01, 0.02],
        }
    )

    result = evaluate_factor(frame, factor_name="neg_factor")
    factor_daily, ls_daily = compute_daily_factor_series(
        frame, factor_name="neg_factor", rev_flag=bool(result["rev_flag"])
    )

    assert result["rev_flag"] is True
    assert (factor_daily["aligned_factor_return"] >= 0).all()
    assert (ls_daily["aligned_long_short_return"] >= 0).all()
