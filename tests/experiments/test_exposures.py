import math
from pathlib import Path

import pandas as pd

from mfe5210_alpha.experiments.exposures import prepare_weighted_exposures
from mfe5210_alpha.model.composite import build_weighted_score


def test_prepare_weighted_exposures_standardizes_each_selected_factor(
    tmp_path: Path,
):
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"] * 4),
            "security": ["A", "B", "C", "D"],
            "fwd_ret_1d": [0.03, 0.01, -0.01, -0.03],
        }
    )
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"] * 4),
            "security": ["A", "B", "C", "D"],
            "f1": [100.0, 10.0, -10.0, -100.0],
        }
    ).to_parquet(tmp_path / "f1.parquet", index=False)

    exposures = prepare_weighted_exposures(
        panel, tmp_path, factor_names=["f1"], factor_signs={"f1": 1}
    )

    assert round(float(exposures["f1"].mean()), 10) == 0.0
    assert round(float(exposures["f1"].std(ddof=0)), 10) == 1.0


def test_build_weighted_score_applies_date_level_weights():
    exposures = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
            "security": ["A", "B"],
            "f1": [1.0, -1.0],
            "f2": [0.5, -0.5],
        }
    )
    weights = pd.DataFrame(
        {"date": pd.to_datetime(["2020-01-02"]), "f1": [0.8], "f2": [0.2]}
    )

    score = build_weighted_score(exposures, weights, ["f1", "f2"])

    assert score.loc[score["security"] == "A", "score"].iloc[0] == 0.9


def test_prepare_weighted_exposures_can_neutralize_by_industry(tmp_path: Path):
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"] * 6),
            "security": list("ABCDEF"),
            "fwd_ret_1d": [0.01, 0.01, 0.01, -0.01, -0.01, -0.01],
            "adv20": [1000.0] * 6,
            "turnover_proxy": [1.0] * 6,
            "member_hs300": [1, 1, 1, 0, 0, 0],
            "member_zz500": [0, 0, 0, 1, 1, 1],
            "industry": ["Bank", "Bank", "Bank", "Tech", "Tech", "Tech"],
        }
    )
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"] * 6),
            "security": list("ABCDEF"),
            "f1": [3.0, 3.1, 2.9, -3.0, -3.1, -2.9],
        }
    ).to_parquet(tmp_path / "f1.parquet", index=False)

    exposures = prepare_weighted_exposures(
        panel,
        tmp_path,
        factor_names=["f1"],
        factor_signs={"f1": 1},
        neutralize=True,
        neutralization_method="ols",
        use_industry=True,
    )

    check = exposures.merge(panel[["date", "security", "industry"]], on=["date", "security"])
    assert check.groupby("industry")["f1"].mean().round(10).abs().max() == 0.0


def test_prepare_weighted_exposures_deduplicates_neutralization_controls(
    tmp_path: Path,
    monkeypatch,
):
    captured = {}

    def capture_neutralize(exposures, controls, factor_names, use_industry=True, method="ols"):
        captured["controls"] = controls.copy()
        return exposures

    monkeypatch.setattr(
        "mfe5210_alpha.experiments.exposures.neutralize_exposures",
        capture_neutralize,
    )

    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"] * 6),
            "security": ["A", "A", "B", "B", "C", "D"],
            "fwd_ret_1d": [0.01, 0.01, 0.02, 0.02, -0.01, -0.02],
            "adv20": [5000.0, 1000.0, 1200.0, 2400.0, 900.0, 800.0],
            "turnover_proxy": [3.0, 1.0, 1.2, 2.4, 0.9, 0.8],
            "member_hs300": [0, 1, 1, 0, 0, 0],
            "member_zz500": [1, 0, 0, 1, 1, 1],
            "industry": ["Asset", "Bank", "Bank", "Bank", "Tech", "Tech"],
        }
    )
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"] * 4),
            "security": list("ABCD"),
            "f1": [2.0, 1.0, -1.0, -2.0],
        }
    ).to_parquet(tmp_path / "f1.parquet", index=False)

    exposures = prepare_weighted_exposures(
        panel,
        tmp_path,
        factor_names=["f1"],
        factor_signs={"f1": 1},
        neutralize=True,
        neutralization_method="ols",
        use_industry=True,
    )

    keys = exposures[["date", "security"]]
    assert not keys.duplicated().any()
    assert len(exposures) == panel[["date", "security"]].drop_duplicates().shape[0]

    control_keys = captured["controls"][["date", "security"]]
    assert not control_keys.duplicated().any()
    a_control = captured["controls"].loc[
        captured["controls"]["security"] == "A"
    ].squeeze()
    assert a_control["industry"] == "Asset"
    assert round(float(a_control["log_adv20"]), 10) == round(math.log1p(5000.0), 10)
    assert a_control["member_hs300"] == 0.0
    assert a_control["member_zz500"] == 1.0
