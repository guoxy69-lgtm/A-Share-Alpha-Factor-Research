import pandas as pd

from mfe5210_alpha.research.oos_selection import select_factors_dynamically


def test_dynamic_admission_rejects_redundant_and_weak_factors():
    summary = pd.DataFrame(
        {
            "factor_name": ["strong", "redundant", "weak"],
            "factor_sign": [1, 1, 1],
            "ls_ann_sharpe": [1.5, 1.4, 0.2],
            "rank_ic_ir": [1.2, 1.1, 0.1],
            "robust_score": [3.0, 2.9, 0.1],
        }
    )
    corr = pd.DataFrame(
        [[1.0, 0.8, 0.1], [0.8, 1.0, 0.1], [0.1, 0.1, 1.0]],
        index=["strong", "redundant", "weak"],
        columns=["strong", "redundant", "weak"],
    )
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"] * 3),
            "factor_name": ["strong", "strong", "redundant", "redundant", "weak", "weak"],
            "aligned_long_short_return": [0.02, 0.02, 0.019, 0.019, 0.001, -0.001],
        }
    )

    selected, details = select_factors_dynamically(
        summary,
        corr,
        daily,
        min_ls_sharpe=0.8,
        min_rank_icir=1.0,
        max_abs_corr=0.5,
        min_incremental_sharpe=0.01,
        max_factors=12,
    )

    assert selected == ["strong"]
    assert (
        details.loc[details["factor_name"] == "redundant", "selection_stage"].iloc[0]
        == "correlation_blocked"
    )
    assert (
        details.loc[details["factor_name"] == "weak", "selection_stage"].iloc[0]
        == "quality_blocked"
    )
