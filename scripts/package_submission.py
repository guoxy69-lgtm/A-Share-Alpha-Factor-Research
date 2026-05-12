from pathlib import Path

from mfe5210_alpha.config import ProjectConfig
from mfe5210_alpha.report.packaging import collect_submission


if __name__ == "__main__":
    cfg = ProjectConfig.from_root(Path(__file__).resolve().parents[1])
    destination = collect_submission(cfg.root, cfg.root / "submission")
    print(destination)
