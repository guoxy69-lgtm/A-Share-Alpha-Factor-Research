from __future__ import annotations

from pathlib import Path

import pandas as pd

from mfe5210_alpha.data.download_daily import filter_a_share_universe


def build_union_membership(hs300: pd.DataFrame, zz500: pd.DataFrame) -> pd.DataFrame:
    hs = filter_a_share_universe(hs300).assign(member_hs300=1, member_zz500=0)
    zz = filter_a_share_universe(zz500).assign(member_hs300=0, member_zz500=1)
    union = pd.concat([hs, zz], ignore_index=True)
    union["date"] = pd.to_datetime(union["date"])
    union = union.rename(columns={"code": "security"})
    return (
        union.groupby(["date", "security"], as_index=False)[
            ["member_hs300", "member_zz500"]
        ]
        .max()
        .sort_values(["date", "security"])
        .reset_index(drop=True)
    )


def _snapshot_query_dates(trade_dates: list[str]) -> list[str]:
    dates = pd.Series(pd.to_datetime(trade_dates), name="date").sort_values()
    if dates.empty:
        return []
    month_ends = dates.groupby(dates.dt.to_period("M")).max()
    query_dates = pd.concat([dates.head(1), month_ends]).drop_duplicates().sort_values()
    return query_dates.dt.strftime("%Y-%m-%d").tolist()


def _assign_effective_date(frame: pd.DataFrame, fallback_date: str) -> pd.DataFrame:
    out = frame.copy()
    if "updateDate" in out.columns and out["updateDate"].notna().any():
        effective_date = str(out["updateDate"].dropna().max())
    else:
        effective_date = fallback_date
    return out.assign(date=effective_date)


def expand_membership_snapshots(
    snapshot_membership: pd.DataFrame, trade_dates: list[str]
) -> pd.DataFrame:
    columns = ["date", "security", "member_hs300", "member_zz500"]
    if snapshot_membership.empty or not trade_dates:
        return pd.DataFrame(columns=columns)

    snapshots = snapshot_membership.copy()
    snapshots["date"] = pd.to_datetime(snapshots["date"])
    trading_days = pd.Series(pd.to_datetime(trade_dates), name="date").sort_values()
    first_trade_date = trading_days.min()
    last_trade_date = trading_days.max()
    effective_dates = [
        date for date in sorted(snapshots["date"].unique()) if date <= last_trade_date
    ]
    frames = []
    for index, effective_date in enumerate(effective_dates):
        next_effective_date = (
            effective_dates[index + 1] if index + 1 < len(effective_dates) else None
        )
        start = max(pd.Timestamp(effective_date), first_trade_date)
        mask = trading_days >= start
        if next_effective_date is not None:
            mask &= trading_days < pd.Timestamp(next_effective_date)
        dates = trading_days.loc[mask]
        if dates.empty:
            continue
        members = snapshots.loc[
            snapshots["date"] == effective_date,
            ["security", "member_hs300", "member_zz500"],
        ].drop_duplicates()
        if members.empty:
            continue
        expanded = members.merge(pd.DataFrame({"date": dates}), how="cross")
        frames.append(expanded[columns])
    if not frames:
        return pd.DataFrame(columns=columns)
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["date", "security"])
        .reset_index(drop=True)
    )


def build_historical_union(client, start_date: str, end_date: str) -> pd.DataFrame:
    trade_dates = client.fetch_trade_dates(start_date, end_date)["date"].tolist()
    hs_frames = []
    zz_frames = []
    for date in _snapshot_query_dates(trade_dates):
        hs_frames.append(_assign_effective_date(client.fetch_index_stocks("hs300", date), date))
        zz_frames.append(_assign_effective_date(client.fetch_index_stocks("zz500", date), date))
    if not hs_frames or not zz_frames:
        return pd.DataFrame(
            columns=["date", "security", "member_hs300", "member_zz500"]
        )
    snapshots = build_union_membership(
        pd.concat(hs_frames, ignore_index=True),
        pd.concat(zz_frames, ignore_index=True),
    )
    return expand_membership_snapshots(snapshots, trade_dates)


def write_universe_history(frame: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return output_path
