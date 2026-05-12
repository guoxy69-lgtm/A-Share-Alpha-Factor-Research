import argparse
from pathlib import Path

from mfe5210_alpha.config import ProjectConfig
from mfe5210_alpha.data.download_daily import download_universe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trade-date", default="2026-04-20")
    parser.add_argument(
        "--universe",
        default="all",
        choices=["all", "sz50", "hs300", "zz500", "hs300_zz500"],
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = ProjectConfig.from_root(Path(__file__).resolve().parents[1])
    for path in download_universe(
        cfg,
        trade_date=args.trade_date,
        universe_name=args.universe,
        limit=args.limit,
        skip_existing=not args.force,
    ):
        print(path)
