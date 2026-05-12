from mfe5210_alpha.factors.catalog import build_factor_catalog


def test_factor_catalog_has_breadth():
    catalog = build_factor_catalog()
    assert len(catalog) >= 60
    assert {"momentum", "volatility", "liquidity", "valuation", "wq_like"} <= {
        item.family for item in catalog
    }


def test_factor_catalog_lists_exact_wq_formulas():
    catalog = {item.name: item for item in build_factor_catalog()}

    assert "see implementation" not in catalog["wq_alpha_001"].formula_text
    assert "mean(pct_change(close, 1), 5)" in catalog["wq_alpha_001"].formula_text
    assert "vwap_proxy" in catalog["wq_alpha_005"].formula_text
