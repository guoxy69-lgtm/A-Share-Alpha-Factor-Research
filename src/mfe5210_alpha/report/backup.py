from __future__ import annotations

import shutil
from pathlib import Path


BACKUP_DIRS = ["report", "report_zh", "submission"]


def create_report_backup(root: Path, backup_root: Path, stamp: str) -> Path:
    destination = backup_root / stamp
    if destination.exists():
        shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True, exist_ok=True)

    for name in BACKUP_DIRS:
        source = root / name
        if source.exists():
            shutil.copytree(source, destination / name, dirs_exist_ok=True)

    return destination
