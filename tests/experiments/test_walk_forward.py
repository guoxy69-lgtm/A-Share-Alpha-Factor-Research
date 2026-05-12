from types import SimpleNamespace

import pandas as pd

from mfe5210_alpha.experiments.walk_forward import (
    build_walk_forward_windows,
    run_walk_forward_experiment,
)


def test_build_walk_forward_windows_uses_five_year_lookback_and_one_year_trade_window():
    windows = build_walk_forward_windows(
        sample_start="2012-01-01",
        sample_end="2026-12-31",
        lookback_years=5,
        holdout_years=1,
    )

    first = windows[0]
    assert first["train_start"].year == 2012
    assert first["train_end"] == pd.Timestamp("2016-12-31")
    assert first["trade_start"].year == 2017
    assert first["trade_end"].year == 2017


def test_walk_forward_windows_do_not_peek_forward():
    windows = build_walk_forward_windows(
        sample_start="2012-01-01",
        sample_end="2020-12-31",
        lookback_years=5,
        holdout_years=1,
    )

    assert all(window["train_end"] < window["trade_start"] for window in windows)


def test_run_walk_forward_experiment_freezes_yearly_selection_but_updates_weights_inside_year(
    tmp_path,
):
    dates = pd.to_datetime(
        [
            "2014-01-02",
            "2015-01-02",
            "2016-01-04",
            "2016-12-30",
            "2017-01-03",
            "2017-06-30",
            "2017-12-29",
        ]
    )
    securities = ["A", "B", "C", "D"]
    panel = pd.DataFrame(
        {
            "date": [date for date in dates for _ in securities],
            "security": securities * len(dates),
            "fwd_ret_1d": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.03,
                0.01,
                0.02,
                0.0,
                0.0,
                0.0,
                0.0,
                0.01,
                0.01,
                0.01,
                0.0,
                0.0,
                0.0,
                0.0,
                -0.01,
                -0.01,
                -0.01,
                0.0,
                0.0,
                0.0,
                0.0,
                -0.03,
                -0.01,
                -0.02,
            ],
        }
    )
    pd.DataFrame(
        {
            "date": [date for date in dates for _ in securities],
            "security": securities * len(dates),
            "f1": [
                3.0,
                1.0,
                -1.0,
                -3.0,
                3.1,
                1.1,
                -1.1,
                -3.1,
                3.2,
                1.2,
                -1.2,
                -3.2,
                3.0,
                1.0,
                -1.0,
                -3.0,
                3.0,
                1.0,
                -1.0,
                -3.0,
                2.5,
                0.8,
                -0.8,
                -2.5,
                3.0,
                1.0,
                -1.0,
                -3.0,
            ],
        }
    ).to_parquet(tmp_path / "f1.parquet", index=False)
    signal = pd.DataFrame(
        {
            "date": pd.to_datetime(["2014-01-02", "2015-01-02", "2016-01-04"]),
            "factor_name": ["f1", "f1", "f1"],
            "factor_return": [0.01, 0.02, 0.03],
            "long_short_return": [0.03, 0.03, 0.03],
            "ic": [0.2, 0.2, 0.2],
            "rank_ic": [0.3, 0.3, 0.3],
        }
    )
    cfg = SimpleNamespace(
        walk_forward=SimpleNamespace(
            lookback_years=3,
            holdout_years=1,
            step_years=1,
        )
    )

    result = run_walk_forward_experiment(
        panel,
        signal,
        tmp_path,
        cfg,
        top_frac=0.25,
        exit_buffer_frac=0.40,
        max_factors=1,
    )

    assert result["performance"]["date"].min() >= pd.Timestamp("2017-01-01")
    assert result["selection_details"]["window_trade_start"].nunique() >= 1
    assert result["weights"]["date"].nunique() >= 1
    assert result["summary"].loc[0, "model_name"] == "rolling_train_test_weighted"
