import pandas as pd

from mfe5210_alpha.research.correlation import compute_research_score, greedy_select_factors


def test_greedy_select_factors_respects_max_abs_corr():
    summary = pd.DataFrame(
        {
            "factor_name": ["f1", "f2", "f3"],
            "research_score": [3.0, 2.0, 1.0],
        }
    )
    corr = pd.DataFrame(
        [[1.0, 0.7, 0.2], [0.7, 1.0, 0.3], [0.2, 0.3, 1.0]],
        index=["f1", "f2", "f3"],
        columns=["f1", "f2", "f3"],
    )

    selected = greedy_select_factors(summary, corr, max_abs_corr=0.5)

    assert selected == ["f1", "f3"]


def test_compute_research_score_uses_aligned_metrics():
    summary = pd.DataFrame(
        {
            "factor_name": ["f1", "f2"],
            "ann_sharpe": [-0.2, 0.1],
            "ls_ann_sharpe": [-0.5, 0.2],
            "ann_ret": [-0.04, 0.01],
            "ls_ann_ret": [-0.08, 0.02],
            "rank_ic_ir": [-0.6, 0.1],
            "monthly_win_rate": [0.7, 0.55],
            "rev_flag": [True, False],
        }
    )

    scored = compute_research_score(summary)

    assert scored.loc[scored["factor_name"] == "f1", "research_score"].iloc[0] > scored.loc[
        scored["factor_name"] == "f2", "research_score"
    ].iloc[0]
