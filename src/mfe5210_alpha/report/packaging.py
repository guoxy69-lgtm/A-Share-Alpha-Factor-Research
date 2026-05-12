from __future__ import annotations

import os
import shutil
from pathlib import Path


TOP_LEVEL_FILES = [
    "README.md",
    "pyproject.toml",
    ".env.example",
]

TOP_LEVEL_DIRS = [
    "src",
    "scripts",
    "tests",
    "report",
    "outputs",
]
REFERENCE_SOURCE_DIR = Path("report") / "reference_files"

SKIP_DIRS = {
    "__pycache__",
    ".pytest_cache",
    "build",
}


def _ignore_names(_directory: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        if name.startswith("._") or name in SKIP_DIRS:
            ignored.add(name)
    return ignored


def _remove_appledouble_files(root: Path) -> None:
    for current_root, dirnames, filenames in os.walk(root):
        current_path = Path(current_root)
        for dirname in list(dirnames):
            if dirname.startswith("._"):
                shutil.rmtree(current_path / dirname, ignore_errors=True)
        for filename in filenames:
            if filename.startswith("._"):
                try:
                    (current_path / filename).unlink()
                except FileNotFoundError:
                    pass


def collect_submission(root: Path, output_root: Path) -> Path:
    root = root.resolve()
    destination = output_root / "MFE5210_AlphaFactors_Guoxy69_Submission"
    if destination.exists():
        shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True, exist_ok=True)

    for file_name in TOP_LEVEL_FILES:
        source = root / file_name
        if source.exists():
            shutil.copy2(source, destination / file_name)

    for directory_name in TOP_LEVEL_DIRS:
        source = root / directory_name
        if not source.exists():
            continue
        target = destination / directory_name
        shutil.copytree(source, target, dirs_exist_ok=True, ignore=_ignore_names)

    reference_source = root / REFERENCE_SOURCE_DIR
    if reference_source.exists():
        shutil.copytree(
            reference_source,
            destination / "references",
            dirs_exist_ok=True,
            ignore=_ignore_names,
        )

    _remove_appledouble_files(destination)
    return destination
