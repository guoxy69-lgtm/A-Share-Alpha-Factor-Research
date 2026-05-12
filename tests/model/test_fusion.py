import pandas as pd
import pytest

from mfe5210_alpha.model.fusion import build_factor_weights


def _summary():
    return pd.DataFrame(
        {
            "factor_name": ["f1", "f2", "f3"],
            "ls_ann_sharpe": [1.2, 0.8, 0.2],
            "rank_ic_ir": [0.5, 1.5, 0.1],
            "ic_mean": [0.02, 0.03, 0.01],
            "robust_score": [2.0, 1.0, 0.1],
        }
    )


def _daily():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"] * 3),
            "factor_name": ["f1"] * 3 + ["f2"] * 3 + ["f3"] * 3,
            "aligned_long_short_return": [
                0.02,
                0.01,
                0.02,
                0.01,
                0.02,
                0.01,
                -0.01,
                0.00,
                -0.01,
            ],
        }
    )


@pytest.mark.parametrize(
    "method",
    [
        "equal_weight",
        "ls_sharpe_weight",
        "rank_icir_weight",
        "ic_mean_weight",
        "robust_score_weight",
        "max_ir",
    ],
)
def test_build_factor_weights_returns_positive_normalized_weights(method):
    weights = build_factor_weights(
        method=method,
        factor_names=["f1", "f2", "f3"],
        summary=_summary(),
        daily_signal_diagnostics=_daily(),
        max_weight=0.70,
    )

    assert set(weights.index) == {"f1", "f2", "f3"}
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0.0).all()
    assert weights.max() <= 0.70 + 1e-12
