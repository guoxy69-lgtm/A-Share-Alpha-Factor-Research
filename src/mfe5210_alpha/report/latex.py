from __future__ import annotations

from pathlib import Path


def build_latex_commands(
    report_dir: Path,
    main_file: str,
    engine: str = "tectonic",
) -> list[list[str]]:
    if Path(engine).stem.lower() == "tectonic":
        return [[engine, main_file]]
    return [
        [engine, "-interaction=nonstopmode", main_file],
        [engine, "-interaction=nonstopmode", main_file],
    ]
