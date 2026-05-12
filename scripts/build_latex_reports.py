from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from mfe5210_alpha.config import ProjectConfig
from mfe5210_alpha.report.latex import build_latex_commands


def _default_engine(root: Path) -> str:
    latest_tectonic = root / ".tools" / "tectonic_latest" / "tectonic.exe"
    if latest_tectonic.exists():
        return str(latest_tectonic)
    local_tectonic = root / ".tools" / "tectonic" / "pkg" / "tools" / "tectonic.exe"
    if local_tectonic.exists():
        return str(local_tectonic)
    return os.environ.get("LATEX_ENGINE", "tectonic")


def _run(report_dir: Path, main_file: str, engine: str, env: dict[str, str] | None = None) -> None:
    for command in build_latex_commands(report_dir, main_file, engine=engine):
        subprocess.run(command, cwd=report_dir, check=True, env=env)


def _copy_engine_to_stage(engine: str, stage: Path) -> str:
    engine_path = Path(engine)
    if not engine_path.exists():
        return engine
    staged_engine = stage / engine_path.name
    shutil.copy2(engine_path, staged_engine)
    return str(staged_engine)


def build_reports(root: Path, engine: str) -> None:
    stage = Path(tempfile.mkdtemp(prefix="mfe5210_latex_build_", dir=Path.home()))
    staged_report = stage / "report"
    staged_report_zh = stage / "report_zh"
    shutil.copytree(root / "report", staged_report)
    shutil.copytree(root / "report_zh", staged_report_zh)

    staged_engine = _copy_engine_to_stage(engine, stage)
    env = os.environ.copy()
    cache_dir = env.get("TECTONIC_CACHE_DIR") or str(stage / ".tectonic-cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    env["TECTONIC_CACHE_DIR"] = cache_dir
    env.setdefault("XDG_CACHE_HOME", cache_dir)

    _run(staged_report, "main.tex", staged_engine, env=env)
    _run(staged_report_zh, "main_zh.tex", staged_engine, env=env)

    shutil.copy2(staged_report / "main.pdf", root / "report" / "main.pdf")
    shutil.copy2(staged_report_zh / "main_zh.pdf", root / "report_zh" / "main_zh.pdf")


if __name__ == "__main__":
    cfg = ProjectConfig.from_root(Path(__file__).resolve().parents[1])
    engine = _default_engine(cfg.root)
    build_reports(cfg.root, engine)
