from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _resolve_project_root(root: Path) -> Path:
    resolved = root.resolve()
    if resolved.parent.name == ".worktrees":
        candidate = resolved.parent.parent
        if (candidate / "data").exists():
            return candidate
    return resolved


@dataclass(frozen=True)
class FixedSplitConfig:
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str


@dataclass(frozen=True)
class WalkForwardConfig:
    lookback_years: int
    step_years: int
    holdout_years: int


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    data_root: Path
    raw_root: Path
    processed_root: Path
    factor_root: Path
    output_root: Path
    report_root: Path
    metadata_db: Path
    sample_start: str
    sample_end: str
    jq_username: Optional[str]
    jq_password: Optional[str]
    universe_name: str
    fixed_split: FixedSplitConfig
    walk_forward: WalkForwardConfig

    @classmethod
    def from_root(cls, root: Path) -> "ProjectConfig":
        root = _resolve_project_root(root)
        data_root = root / "data"
        metadata_root = data_root / "metadata"
        return cls(
            root=root,
            data_root=data_root,
            raw_root=data_root / "raw",
            processed_root=data_root / "processed",
            factor_root=data_root / "factor_store",
            output_root=root / "outputs",
            report_root=root / "report",
            metadata_db=metadata_root / "catalog.sqlite",
            sample_start="2012-01-01",
            sample_end="2026-04-20",
            jq_username=os.environ.get("JQDATA_USERNAME"),
            jq_password=os.environ.get("JQDATA_PASSWORD"),
            universe_name="hs300_zz500",
            fixed_split=FixedSplitConfig(
                train_end="2021-12-31",
                validation_start="2019-01-01",
                validation_end="2021-12-31",
                test_start="2022-01-01",
            ),
            walk_forward=WalkForwardConfig(
                lookback_years=5,
                step_years=1,
                holdout_years=1,
            ),
        )
