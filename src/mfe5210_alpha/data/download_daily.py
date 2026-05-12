from __future__ import annotations

from pathlib import Path

import pandas as pd

from mfe5210_alpha.config import ProjectConfig
from mfe5210_alpha.data.baostock_client import BaostockClient

A_SHARE_PREFIXES = (
    "sh.600",
    "sh.601",
    "sh.603",
    "sh.605",
    "sh.688",
    "sz.000",
    "sz.001",
    "sz.002",
    "sz.003",
    "sz.300",
    "sz.301",
)

PRICE_FIELDS = [
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "amount",
    "adjustflag",
    "turn",
    "tradestatus",
    "pctChg",
    "peTTM",
    "pbMRQ",
    "psTTM",
    "pcfNcfTTM",
    "isST",
]


def filter_a_share_universe(universe: pd.DataFrame) -> pd.DataFrame:
    mask = universe["code"].astype(str).str.startswith(A_SHARE_PREFIXES)
    return universe.loc[mask].copy().reset_index(drop=True)


def save_price_frame(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)


def price_output_path(cfg: ProjectConfig, security: str) -> Path:
    return cfg.raw_root / "prices" / f"{security}.parquet"


def _next_date_string(date_value: object) -> str:
    return (pd.to_datetime(date_value) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def _existing_price_frame(output_path: Path) -> pd.DataFrame | None:
    if not output_path.exists():
        return None
    try:
        frame = pd.read_parquet(output_path)
    except Exception:
        return None
    if "date" not in frame.columns:
        return None
    return frame


def price_file_is_current(output_path: Path, sample_end: str) -> bool:
    frame = _existing_price_frame(output_path)
    if frame is None or frame.empty:
        return False
    max_date = pd.to_datetime(frame["date"], errors="coerce").max()
    return bool(pd.notna(max_date) and max_date >= pd.Timestamp(sample_end))


def load_universe(
    client: BaostockClient, trade_date: str, universe_name: str = "all"
) -> pd.DataFrame:
    if universe_name == "all":
        return filter_a_share_universe(client.fetch_all_stocks(trade_date))
    if universe_name in {"sz50", "hs300", "zz500"}:
        return filter_a_share_universe(client.fetch_index_stocks(universe_name, trade_date))
    if universe_name == "hs300_zz500":
        combined = pd.concat(
            [
                client.fetch_index_stocks("hs300", trade_date),
                client.fetch_index_stocks("zz500", trade_date),
            ],
            ignore_index=True,
        )
        return (
            filter_a_share_universe(combined)
            .drop_duplicates(subset=["code"])
            .reset_index(drop=True)
        )
    raise ValueError(f"unsupported universe: {universe_name}")


def download_one_security(
    cfg: ProjectConfig, security: str, client: BaostockClient | None = None
) -> Path:
    client = client or BaostockClient()
    output_path = price_output_path(cfg, security)
    existing = _existing_price_frame(output_path)
    start_date = cfg.sample_start
    if existing is not None and not existing.empty:
        max_date = pd.to_datetime(existing["date"], errors="coerce").max()
        if pd.notna(max_date) and max_date >= pd.Timestamp(cfg.sample_end):
            return output_path
        if pd.notna(max_date):
            start_date = _next_date_string(max_date)

    frame = client.fetch_price(security, start_date, cfg.sample_end, PRICE_FIELDS)
    if existing is not None and not existing.empty:
        frame = pd.concat([existing, frame], ignore_index=True)
        dedupe_cols = [column for column in ["date", "code"] if column in frame.columns]
        if dedupe_cols:
            frame = frame.drop_duplicates(subset=dedupe_cols, keep="last")
        frame = frame.sort_values("date").reset_index(drop=True)
    save_price_frame(frame, output_path)
    return output_path


def download_universe(
    cfg: ProjectConfig,
    trade_date: str,
    universe_name: str = "all",
    limit: int | None = None,
    client: BaostockClient | None = None,
    downloader=download_one_security,
    skip_existing: bool = True,
) -> list[Path]:
    client = client or BaostockClient()
    universe = load_universe(client, trade_date, universe_name)
    securities = universe["code"].tolist()
    if limit is not None:
        securities = securities[:limit]
    output_paths = []
    for security in securities:
        existing_path = price_output_path(cfg, security)
        if skip_existing and price_file_is_current(existing_path, cfg.sample_end):
            output_paths.append(existing_path)
            continue
        output_paths.append(downloader(cfg, security, client=client))
    return output_paths
