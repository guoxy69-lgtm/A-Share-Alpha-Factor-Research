from pathlib import Path
import argparse

import pandas as pd

from mfe5210_alpha.config import ProjectConfig
from mfe5210_alpha.data.build_research_store import build_research_panel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe-csv", default=None)
    args = parser.parse_args()

    cfg = ProjectConfig.from_root(Path(__file__).resolve().parents[1])
    allowed_securities = None
    if args.universe_csv:
        universe = pd.read_csv(args.universe_csv)
        allowed_securities = set(universe["code"].astype(str))

    price_dir = cfg.raw_root / "prices"
    parquet_paths = [
        path for path in sorted(price_dir.glob("*.parquet")) if not path.name.startswith("._")
    ]
    if allowed_securities is not None:
        parquet_paths = [
            path for path in parquet_paths if path.stem in allowed_securities
        ]
    frames = [pd.read_parquet(path) for path in parquet_paths]
    panel = build_research_panel(
        pd.concat(frames, ignore_index=True), allowed_securities=allowed_securities
    )
    output_path = cfg.processed_root / "research_panel.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_path, index=False)
    print(output_path)
