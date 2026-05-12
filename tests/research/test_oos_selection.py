import pandas as pd

from mfe5210_alpha.research.oos_selection import (
    compute_equal_weight_robust_score,
    determine_factor_sign,
    select_factors_for_window,
    summarize_window,
)


def test_determine_factor_sign_requires_consistent_rankic_and_spread():
    metrics = pd.DataFrame(
        [
            {"factor_name": "f1", "rank_ic_mean": 0.02, "ls_ann_ret": 0.10},
            {"factor_name": "f2", "rank_ic_mean": -0.01, "ls_ann_ret": -0.08},
            {"factor_name": "f3", "rank_ic_mean": 0.02, "ls_ann_ret": -0.01},
            {"factor_name": "f4", "rank_ic_mean": 0.00, "ls_ann_ret": 0.01},
        ]
    )

    assert determine_factor_sign(metrics.iloc[0]) == 1
    assert determine_factor_sign(metrics.iloc[1]) == -1
    assert determine_factor_sign(metrics.iloc[2]) is None
    assert determine_factor_sign(metrics.iloc[3]) is None


def test_equal_weight_robust_score_uses_all_available_components():
    summary = pd.DataFrame(
        {
            "factor_name": ["stable", "weak"],
            "ls_ann_sharpe": [1.2, 0.2],
            "rank_ic_ir": [1.0, 0.1],
            "ic_ir": [0.8, 0.1],
            "monthly_win_rate": [0.7, 0.4],
            "recent_ls_sharpe": [1.1, 0.1],
            "factor_max_drawdown": [-0.05, -0.40],
            "rank_ic_mean": [0.03, 0.01],
            "ls_ann_ret": [0.12, 0.02],
        }
    )

    scored = compute_equal_weight_robust_score(summary)

    assert scored.loc[scored["factor_name"] == "stable", "robust_score"].iloc[0] > scored.loc[
        scored["factor_name"] == "weak", "robust_score"
    ].iloc[0]
    assert scored["factor_sign"].tolist() == [1, 1]


def test_summarize_window_derives_factor_sign_from_window_only():
    diagnostics = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-02", "2020-01-03", "2020-02-03", "2020-02-04"]
            ),
            "factor_name": ["reversal", "reversal", "reversal", "reversal"],
            "factor_return": [-0.01, -0.02, -0.01, -0.02],
            "long_short_return": [-0.01, -0.02, -0.01, -0.02],
            "ic": [-0.2, -0.1, -0.2, -0.1],
            "rank_ic": [-0.3, -0.2, -0.3, -0.2],
        }
    )

    summary = summarize_window(diagnostics)

    assert summary.loc[0, "factor_name"] == "reversal"
    assert summary.loc[0, "factor_sign"] == -1
    assert summary.loc[0, "raw_ls_ann_ret"] < 0
    assert summary.loc[0, "ls_ann_ret"] > 0
    assert summary.loc[0, "ls_ann_sharpe"] > 0
    assert summary.loc[0, "monthly_win_rate"] == 1.0
    assert summary.loc[0, "recent_ls_sharpe"] > 0


def test_select_factors_for_window_enforces_abs_corr_limit():
    summary = pd.DataFrame(
        {
            "factor_name": ["f1", "f2", "f3", "unstable"],
            "robust_score": [2.0, 1.8, 1.0, 3.0],
            "factor_sign": [1, 1, 1, None],
        }
    )
    corr = pd.DataFrame(
        [
            [1.0, 0.6, 0.2, 0.0],
            [0.6, 1.0, 0.1, 0.0],
            [0.2, 0.1, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        index=["f1", "f2", "f3", "unstable"],
        columns=["f1", "f2", "f3", "unstable"],
    )

    selected, details = select_factors_for_window(summary, corr, max_abs_corr=0.5)

    assert selected == ["f1", "f3"]
    assert details.loc[details["factor_name"] == "f2", "selection_stage"].iloc[
        0
    ] == "correlation_blocked"
    assert details.loc[details["factor_name"] == "unstable", "selection_stage"].iloc[
        0
    ] == "dropped_unstable_sign"
