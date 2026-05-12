from types import SimpleNamespace

import pandas as pd

from mfe5210_alpha.experiments.fusion_grid import (
    ExperimentSpec,
    build_method_specs,
    run_fixed_fusion_spec,
    run_walk_forward_fusion_spec,
)


def test_build_method_specs_includes_core_industry_methods():
    specs = build_method_specs()
    names = {spec.name for spec in specs}

    assert "fixed_rank_icir_ls20_neutral_ols" in names
    assert "fixed_max_ir_ls20_neutral_ols" in names
    assert "fixed_equal_ls20_raw" in names
    assert all(isinstance(spec, ExperimentSpec) for spec in specs)


def test_run_fixed_fusion_spec_uses_training_window_only(tmp_path):
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
            "adv20": [1000.0] * 12,
            "turnover_proxy": [1.0] * 12,
            "member_hs300": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            "member_zz500": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            "industry": [
                "Bank",
                "Bank",
                "Bank",
                "Bank",
                "Bank",
                "Bank",
                "Tech",
                "Tech",
                "Tech",
                "Tech",
                "Tech",
                "Tech",
            ],
        }
    )
    signal = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2018-01-02", "2018-01-03", "2018-01-04", "2018-01-05"]
            ),
            "factor_name": ["f1"] * 4,
            "factor_return": [0.02, 0.01, 0.03, 0.02],
            "long_short_return": [0.03, 0.02, 0.04, 0.01],
            "ic": [0.2, 0.1, 0.3, 0.2],
            "rank_ic": [0.3, 0.2, 0.4, 0.25],
        }
    )
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-01-02", "2022-01-04", "2022-01-05"] * 4),
            "security": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "D", "D", "D"],
            "f1": [3.0, 3.0, 3.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -3.0, -3.0, -3.0],
        }
    ).to_parquet(tmp_path / "f1.parquet", index=False)

    spec = ExperimentSpec(
        name="fixed_equal_ls20_raw",
        route="fixed",
        fusion_method="equal_weight",
        top_frac=0.25,
        exit_buffer_frac=0.40,
        neutralize=False,
        neutralization_method="ols",
        use_industry=False,
        portfolio_type="long_short",
    )
    cfg = SimpleNamespace(
        fixed_split=SimpleNamespace(train_end="2021-12-31", test_start="2022-01-01")
    )

    result = run_fixed_fusion_spec(panel, signal, tmp_path, cfg, spec)

    assert result["performance"]["date"].min() >= pd.Timestamp("2022-01-01")
    assert not result["summary"].empty
    assert result["weights"].filter(regex="^f").sum(axis=1).round(10).eq(1.0).all()


def test_run_walk_forward_fusion_spec_uses_past_year_window(tmp_path):
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2021-01-04", "2021-01-05"] * 4),
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
            "adv20": [1000.0] * 12,
            "turnover_proxy": [1.0] * 12,
            "member_hs300": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            "member_zz500": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        }
    )
    signal = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
            ),
            "factor_name": ["f1"] * 4,
            "factor_return": [0.02, 0.01, 0.03, 0.02],
            "long_short_return": [0.03, 0.02, 0.04, 0.01],
            "ic": [0.2, 0.1, 0.3, 0.2],
            "rank_ic": [0.3, 0.2, 0.4, 0.25],
        }
    )
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2021-01-04", "2021-01-05"] * 4),
            "security": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "D", "D", "D"],
            "f1": [3.0, 3.0, 3.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -3.0, -3.0, -3.0],
        }
    ).to_parquet(tmp_path / "f1.parquet", index=False)

    spec = ExperimentSpec(
        name="rolling_equal_ls20_raw",
        route="rolling",
        fusion_method="equal_weight",
        top_frac=0.25,
        exit_buffer_frac=0.40,
        neutralize=False,
        neutralization_method="ols",
        use_industry=False,
        portfolio_type="long_short",
    )
    cfg = SimpleNamespace(
        walk_forward=SimpleNamespace(lookback_years=1, holdout_years=1, step_years=1)
    )

    result = run_walk_forward_fusion_spec(panel, signal, tmp_path, cfg, spec)

    assert result["performance"]["date"].min() >= pd.Timestamp("2021-01-01")
    assert not result["summary"].empty
    assert result["weights"].filter(regex="^f").sum(axis=1).round(10).eq(1.0).all()
