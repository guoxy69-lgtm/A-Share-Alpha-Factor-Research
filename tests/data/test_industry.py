import pandas as pd

from mfe5210_alpha.data.industry import attach_industry_classification, normalize_baostock_industry


def test_normalize_baostock_industry_standardizes_schema():
    raw = pd.DataFrame(
        {
            "code": ["sh.600000", "sz.000001"],
            "code_name": ["浦发银行", "平安银行"],
            "industry": ["银行", "银行"],
            "industryClassification": ["申万一级行业", "申万一级行业"],
            "asof_date": ["2026-04-24", "2026-04-24"],
            "source": ["baostock.query_stock_industry", "baostock.query_stock_industry"],
        }
    )

    out = normalize_baostock_industry(raw)

    assert list(out.columns) == [
        "security",
        "code_name",
        "industry",
        "industry_classification",
        "asof_date",
        "source",
    ]
    assert out["security"].tolist() == ["sh.600000", "sz.000001"]
    assert str(out["asof_date"].iloc[0].date()) == "2026-04-24"


def test_attach_industry_classification_uses_static_mapping():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-04", "2022-01-04"]),
            "security": ["sh.600000", "sz.000001"],
            "fwd_ret_1d": [0.01, -0.01],
        }
    )
    industry = pd.DataFrame(
        {
            "security": ["sh.600000", "sz.000001"],
            "code_name": ["浦发银行", "平安银行"],
            "industry": ["银行", "银行"],
            "industry_classification": ["申万一级行业", "申万一级行业"],
            "asof_date": pd.to_datetime(["2026-04-24", "2026-04-24"]),
            "source": ["baostock.query_stock_industry", "baostock.query_stock_industry"],
        }
    )

    out = attach_industry_classification(panel, industry)

    assert "industry" in out.columns
    assert out["code_name"].tolist() == ["浦发银行", "平安银行"]
    assert out["industry"].eq("银行").all()
    assert out["industry_is_static"].eq(True).all()
