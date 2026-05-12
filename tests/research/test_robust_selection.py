import pandas as pd

from mfe5210_alpha.research.robust_selection import (
    assign_correlation_clusters,
    compute_robust_factor_score,
    select_production_factors,
)


def test_assign_correlation_clusters_groups_highly_related_factors():
    corr = pd.DataFrame(
        [
            [1.0, 0.82, 0.20],
            [0.82, 1.0, 0.10],
            [0.20, 0.10, 1.0],
        ],
        index=["f1", "f2", "f3"],
        columns=["f1", "f2", "f3"],
    )

    clusters = assign_correlation_clusters(corr, cluster_abs_corr=0.7)
    by_name = clusters.set_index("factor_name")["cluster_id"].to_dict()

    assert by_name["f1"] == by_name["f2"]
    assert by_name["f3"] != by_name["f1"]


def test_compute_robust_factor_score_rewards_recent_stability_and_penalizes_drawdown():
    summary = pd.DataFrame(
        {
            "factor_name": ["stable", "fragile"],
            "ann_sharpe": [1.0, 1.0],
            "ls_ann_sharpe": [1.5, 1.5],
            "rank_ic_ir": [1.2, 1.2],
            "ic_ir": [1.1, 1.1],
            "ls_ann_ret": [0.20, 0.20],
            "monthly_win_rate": [0.60, 0.60],
        }
    )
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-16", "2026-04-17", "2026-04-20"] * 2),
            "factor_name": ["stable"] * 3 + ["fragile"] * 3,
            "aligned_long_short_return": [0.01, 0.01, 0.01, 0.20, -0.30, 0.01],
        }
    )

    scored = compute_robust_factor_score(summary, daily, recent_window=3)
    score = scored.set_index("factor_name")["robust_score"]

    assert score["stable"] > score["fragile"]
    assert scored.loc[scored["factor_name"] == "fragile", "factor_max_drawdown"].iloc[0] < 0


def test_select_production_factors_uses_cluster_representatives_then_final_corr_guard():
    summary = pd.DataFrame(
        {
            "factor_name": ["f1", "f2", "f3"],
            "ann_sharpe": [2.0, 1.9, 1.0],
            "ls_ann_sharpe": [2.0, 1.9, 1.1],
            "rank_ic_ir": [1.5, 1.4, 1.0],
            "ic_ir": [1.4, 1.3, 0.9],
            "ls_ann_ret": [0.2, 0.18, 0.12],
            "monthly_win_rate": [0.7, 0.68, 0.62],
        }
    )
    corr = pd.DataFrame(
        [
            [1.0, 0.86, 0.30],
            [0.86, 1.0, 0.20],
            [0.30, 0.20, 1.0],
        ],
        index=["f1", "f2", "f3"],
        columns=["f1", "f2", "f3"],
    )

    selected, details = select_production_factors(
        summary,
        corr,
        max_abs_corr=0.5,
        cluster_abs_corr=0.7,
    )

    assert selected == ["f1", "f3"]
    assert details.loc[details["factor_name"] == "f1", "selection_stage"].iloc[0] == "selected"
    assert details.loc[details["factor_name"] == "f2", "selection_stage"].iloc[0] == "cluster_redundant"
