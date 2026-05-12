from pathlib import Path

from mfe5210_alpha.report.packaging import collect_submission


def test_collect_submission_copies_code_report_and_outputs(tmp_path: Path):
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "report" / "assets").mkdir(parents=True)
    (root / "outputs" / "selection").mkdir(parents=True)
    (root / "data" / "raw").mkdir(parents=True)
    (root / "src" / "module.py").write_text("print('ok')", encoding="utf-8")
    (root / "scripts" / "run.py").write_text("print('run')", encoding="utf-8")
    (root / "report" / "main.pdf").write_text("pdf", encoding="utf-8")
    (root / "outputs" / "selection" / "selected_factors.csv").write_text("x", encoding="utf-8")
    (root / "README.md").write_text("readme", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]", encoding="utf-8")
    (root / "data" / "raw" / "large.parquet").write_text("skip", encoding="utf-8")
    (root / "._junk").write_text("skip", encoding="utf-8")

    destination = collect_submission(root, tmp_path / "submission")

    assert (destination / "src" / "module.py").exists()
    assert (destination / "scripts" / "run.py").exists()
    assert (destination / "report" / "main.pdf").exists()
    assert (destination / "outputs" / "selection" / "selected_factors.csv").exists()
    assert not (destination / "data" / "raw" / "large.parquet").exists()
    assert not (destination / "._junk").exists()


def test_collect_submission_rebuilds_destination_without_stale_appledouble_files(tmp_path: Path):
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "src" / "module.py").write_text("print('ok')", encoding="utf-8")
    destination = tmp_path / "submission" / "MFE5210_AlphaFactors_Guoxy69_Submission"
    destination.mkdir(parents=True)
    (destination / "._stale").write_text("stale", encoding="utf-8")

    collect_submission(root, tmp_path / "submission")

    assert not (destination / "._stale").exists()


def test_collect_submission_copies_reference_bundle(tmp_path: Path):
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "report" / "reference_files").mkdir(parents=True)
    (root / "report" / "reference_files" / "worldquant_101_alphas.pdf").write_text(
        "pdf",
        encoding="utf-8",
    )

    destination = collect_submission(root, tmp_path / "submission")

    assert (destination / "references" / "worldquant_101_alphas.pdf").exists()
