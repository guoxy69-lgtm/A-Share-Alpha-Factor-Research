from pathlib import Path

from mfe5210_alpha.config import ProjectConfig
from mfe5210_alpha.data.baostock_client import BaostockClient
from mfe5210_alpha.data.universe_history import (
    build_historical_union,
    write_universe_history,
)


if __name__ == "__main__":
    cfg = ProjectConfig.from_root(Path(__file__).resolve().parents[1])
    client = BaostockClient()
    frame = build_historical_union(client, cfg.sample_start, cfg.sample_end)
    print(
        write_universe_history(
            frame, cfg.raw_root / "universe" / "hs300_zz500_history.parquet"
        )
    )
