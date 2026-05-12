import pandas as pd

from mfe5210_alpha.report.appendices import (
    extract_formula_fields,
    operator_definitions,
    write_correlation_matrix,
    write_factor_dictionary,
    write_single_factor_full_results,
)


def test_extract_formula_fields_finds_project_data_fields():
    fields = extract_formula_fields("mean((close-open)/open, 5) + zscore(pbMRQ)")

    assert fields == "close, open, pbMRQ"


def test_operator_definitions_include_delay_and_rankic():
    definitions = operator_definitions()

    assert {"delay", "rankIC"} <= set(definitions["operator"])


def test_write_factor_dictionary_contains_formula_and_fields(tmp_path):
    catalog = pd.DataFrame(
        {
            "factor_name": ["return_5d"],
            "family": ["momentum"],
            "description": ["5-day return"],
            "formula_text": ["close / delay(close, 5) - 1"],
        }
    )

    stats = pd.DataFrame({"factor_name": ["return_5d"], "aligned_ann_sharpe": [1.234]})

    output = write_factor_dictionary(
        catalog,
        tmp_path / "factor_dictionary.tex",
        factor_stats=stats,
    )
    text = output.read_text(encoding="utf-8")

    assert "return\\_5d" in text
    assert "close" in text
    assert "delay" in text
    assert "Aligned Sharpe" in text
    assert "1.234" in text


def test_write_single_factor_full_results_lists_all_rows(tmp_path):
    summary = pd.DataFrame(
        {
            "factor_name": ["f1", "f2"],
            "ann_sharpe": [1.1, -0.2],
            "ls_ann_sharpe": [1.2, -0.3],
            "ic_mean": [0.01, -0.01],
            "rank_ic_mean": [0.02, -0.02],
            "ic_ir": [0.5, -0.1],
            "rank_ic_ir": [0.6, -0.2],
            "monthly_win_rate": [0.7, 0.4],
            "rev_flag": [False, True],
        }
    )

    output = write_single_factor_full_results(summary, tmp_path / "single_factor.tex")
    text = output.read_text(encoding="utf-8")

    assert "f1" in text
    assert "f2" in text
    assert "Yes" in text


def test_write_correlation_matrix_outputs_square_table(tmp_path):
    matrix = pd.DataFrame(
        [[1.0, 0.2], [0.2, 1.0]],
        index=["factor_a", "factor_b"],
        columns=["factor_a", "factor_b"],
    )

    output = write_correlation_matrix(matrix, tmp_path / "corr.tex")
    text = output.read_text(encoding="utf-8")

    assert "factor\\_a" in text
    assert "0.200" in text
