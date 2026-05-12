from pathlib import Path

from mfe5210_alpha.report.backup import create_report_backup


def test_create_report_backup_copies_report_trees_and_submission(tmp_path: Path):
    root = tmp_path / "project"
    (root / "report").mkdir(parents=True)
    (root / "report_zh").mkdir()
    (root / "submission" / "final").mkdir(parents=True)
    (root / "report" / "main.tex").write_text("english", encoding="utf-8")
    (root / "report_zh" / "main_zh.tex").write_text("chinese", encoding="utf-8")
    (root / "submission" / "final" / "README.md").write_text("package", encoding="utf-8")

    backup = create_report_backup(root, tmp_path / "report_backups", stamp="2026-04-28_120000")

    assert (backup / "report" / "main.tex").read_text(encoding="utf-8") == "english"
    assert (backup / "report_zh" / "main_zh.tex").read_text(encoding="utf-8") == "chinese"
    assert (backup / "submission" / "final" / "README.md").read_text(encoding="utf-8") == "package"
