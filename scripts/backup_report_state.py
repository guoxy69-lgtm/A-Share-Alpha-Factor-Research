from __future__ import annotations

from datetime import datetime
from pathlib import Path

from mfe5210_alpha.config import ProjectConfig
from mfe5210_alpha.report.backup import create_report_backup


if __name__ == "__main__":
    cfg = ProjectConfig.from_root(Path(__file__).resolve().parents[1])
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    backup = create_report_backup(cfg.root, cfg.root / "report_backups", stamp)
    print(backup)
