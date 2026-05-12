import pandas as pd

from mfe5210_alpha.report.assets import build_summary_rows, latex_escape


def test_latex_escape_handles_special_characters():
    assert latex_escape("wq_alpha_001 & turnover") == "wq\\_alpha\\_001 \\& turnover"


def test_build_summary_rows_combines_core_outputs():
    selected = pd.DataFrame({"selected_factor_count": [12], "max_abs_corr": [0.458]})
    strategy = pd.DataFrame(
        {
            "annualized_return": [0.082],
            "annualized_sharpe": [0.399],
            "max_drawdown": [-0.440],
            "average_one_way_turnover": [0.648],
        }
    )
    single = pd.DataFrame({"ann_sharpe": [0.1, 0.2, -0.1]})

    rows = build_summary_rows(selected, strategy, single)

    assert ("Selected factor count", "12") in rows
    assert ("Maximum absolute factor correlation", "0.458") in rows
    assert ("Average single-factor Sharpe", "0.067") in rows


def test_build_summary_rows_can_select_primary_model():
    selected = pd.DataFrame({"selected_factor_count": [12], "max_abs_corr": [0.458]})
    strategy = pd.DataFrame(
        {
            "model_name": ["daily", "smoothed"],
            "annualized_return": [0.50, 0.40],
            "annualized_sharpe": [3.20, 3.05],
            "max_drawdown": [-0.22, -0.18],
            "average_one_way_turnover": [1.20, 0.83],
        }
    )
    single = pd.DataFrame({"ann_sharpe": [0.1, 0.2, -0.1]})

    rows = build_summary_rows(selected, strategy, single, primary_model_name="smoothed")

    assert ("Reported model", "smoothed") in rows
    assert ("Maximum drawdown", "-0.180") in rows
