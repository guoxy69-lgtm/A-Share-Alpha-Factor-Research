from types import SimpleNamespace

import pandas as pd
import pytest

from mfe5210_alpha.experiments.fixed_split import (
    build_fixed_split_windows,
    run_fixed_split_experiment,
)
from scripts.build_oos_comparison import summarize_experiment


def test_build_fixed_split_windows_keeps_test_period_untouched():
    windows = build_fixed_split_windows(
        sample_start="2012-01-01",
        sample_end="2026-04-20",
        train_end="2021-12-31",
        test_start="2022-01-01",
    )

    assert windows["train"][1] == pd.Timestamp("2021-12-31")
    assert windows["test"][0] == pd.Timestamp("2022-01-01")


def test_run_fixed_split_experiment_freezes_training_outputs_for_test_period(
    tmp_path,
):
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-01-02", "2022-01-04", "2022-01-05"] * 4),
            "security": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "D", "D", "D"],
            "fwd_ret_1d": [
                0.0,
                0.03,
                0.02,
                0.0,
                0.01,
                0.01,
                0.0,
                -0.01,
                -0.01,
                0.0,
                -0.03,
                -0.02,
            ],
        }
    )
    signal = pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-01-02", "2018-01-03"]),
            "factor_name": ["f1", "f1"],
            "factor_return": [0.01, 0.02],
            "long_short_return": [0.03, 0.03],
            "ic": [0.2, 0.2],
            "rank_ic": [0.3, 0.3],
        }
    )
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-01-02", "2022-01-04", "2022-01-05"] * 4),
            "security": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "D", "D", "D"],
            "f1": [3.0, 3.0, 3.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -3.0, -3.0, -3.0],
        }
    ).to_parquet(tmp_path / "f1.parquet", index=False)

    cfg = SimpleNamespace(
        fixed_split=SimpleNamespace(train_end="2021-12-31", test_start="2022-01-01")
    )
    result = run_fixed_split_experiment(
        panel,
        signal,
        tmp_path,
        cfg,
        top_frac=0.25,
        exit_buffer_frac=0.40,
        max_factors=1,
    )

    assert result["performance"]["date"].min() >= pd.Timestamp("2022-01-01")
    assert result["selection_details"]["selection_stage"].eq("selected").any()
    assert "weights" in result


def test_summarize_experiment_uses_cagr_and_final_nav():
    perf = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-04", "2022-01-05"]),
            "portfolio_return": [0.01, -0.01],
        }
    )
    summary = pd.DataFrame(
        {
            "annualized_return": [0.12],
            "annualized_sharpe": [1.5],
            "max_drawdown": [-0.08],
            "average_one_way_turnover": [0.2],
        }
    )

    row = summarize_experiment(perf, summary, "fixed_train_test")
    expected_nav = (1.0 + perf["portfolio_return"]).prod()
    expected_cagr = expected_nav ** (243 / len(perf)) - 1.0

    assert row["final_nav"] == pytest.approx(expected_nav)
    assert row["cagr"] == pytest.approx(expected_cagr)
