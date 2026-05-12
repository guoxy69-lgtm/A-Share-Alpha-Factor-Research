from mfe5210_alpha.report.references import default_references, render_bibtex


def test_default_references_include_worldquant_and_broker_reports():
    references = default_references()
    keys = {reference.key for reference in references}

    assert "worldquant_101_alphas" in keys
    assert "gtja_short_cycle_price_volume_2017" in keys
    assert "huatai_momentum_2016" in keys


def test_render_bibtex_outputs_misc_entries():
    bibtex = render_bibtex(default_references())

    assert "@misc{worldquant_101_alphas" in bibtex
    assert "WorldQuant 101 Formulaic Alphas" in bibtex
    assert "数量化专题之九十三：基于短周期价量特征的多因子选股体系" in bibtex
    assert "国泰君安证券" in bibtex


def test_default_references_include_source_files_for_bundling():
    sources = {reference.key: reference.source_file for reference in default_references()}

    assert sources["worldquant_101_alphas"] == "worldquant_101_alphas.pdf"
    assert sources["gtja_short_cycle_price_volume_2017"].endswith(".pdf")
