from pathlib import Path

from mfe5210_alpha.config import ProjectConfig


def test_project_config_builds_expected_paths(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("JQDATA_USERNAME", "dummy_user")
    monkeypatch.setenv("JQDATA_PASSWORD", "dummy_password")

    cfg = ProjectConfig.from_root(tmp_path)

    assert cfg.sample_start == "2012-01-01"
    assert cfg.sample_end == "2026-04-20"
    assert cfg.data_root == tmp_path / "data"
    assert cfg.raw_root == tmp_path / "data" / "raw"
    assert cfg.processed_root == tmp_path / "data" / "processed"
    assert cfg.factor_root == tmp_path / "data" / "factor_store"
    assert cfg.metadata_db == tmp_path / "data" / "metadata" / "catalog.sqlite"
    assert cfg.universe_name == "hs300_zz500"
    assert cfg.fixed_split.train_end == "2021-12-31"
    assert cfg.fixed_split.validation_start == "2019-01-01"
    assert cfg.fixed_split.validation_end == "2021-12-31"
    assert cfg.fixed_split.test_start == "2022-01-01"
    assert cfg.walk_forward.lookback_years == 5
    assert cfg.walk_forward.step_years == 1
    assert cfg.walk_forward.holdout_years == 1


def test_project_config_resolves_shared_project_root_from_worktree(tmp_path: Path):
    project_root = tmp_path / "project"
    worktree_root = project_root / ".worktrees" / "feature-branch"
    (project_root / "data").mkdir(parents=True)
    worktree_root.mkdir(parents=True)

    cfg = ProjectConfig.from_root(worktree_root)

    assert cfg.root == project_root
    assert cfg.data_root == project_root / "data"
    assert cfg.output_root == project_root / "outputs"
