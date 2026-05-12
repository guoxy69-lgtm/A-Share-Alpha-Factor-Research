from __future__ import annotations

import pandas as pd

from mfe5210_alpha.factors.base import output_factor, sort_panel


def _make_wq_alpha_001(frame: pd.DataFrame) -> pd.DataFrame:
    out = sort_panel(frame)
    values = (
        out.groupby("security")["close"]
        .pct_change(1, fill_method=None)
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return output_factor(out, "wq_alpha_001", values)


def _make_wq_alpha_002(frame: pd.DataFrame) -> pd.DataFrame:
    out = sort_panel(frame)
    values = ((out["high"] + out["low"]) / 2 - out["close"]) / out["close"].replace(0, pd.NA)
    return output_factor(out, "wq_alpha_002", values)


def _make_wq_alpha_003(frame: pd.DataFrame) -> pd.DataFrame:
    out = sort_panel(frame)
    values = ((out["close"] - out["open"]) / out["open"].replace(0, pd.NA)) * out["turn"]
    return output_factor(out, "wq_alpha_003", values)


def _make_wq_alpha_004(frame: pd.DataFrame) -> pd.DataFrame:
    out = sort_panel(frame)
    values = out.groupby("security")["amount"].pct_change(5, fill_method=None)
    return output_factor(out, "wq_alpha_004", values)


def factor_wq_alpha_005(frame: pd.DataFrame) -> pd.DataFrame:
    out = sort_panel(frame)
    vwap_proxy = (out["high"] + out["low"] + out["close"]) / 3.0
    values = (out["open"] - vwap_proxy.groupby(out["security"]).rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)) * (
        out["close"] - vwap_proxy
    ).abs()
    return output_factor(out, "wq_alpha_005", values)


def _make_wq_alpha_006(frame: pd.DataFrame) -> pd.DataFrame:
    out = sort_panel(frame)
    values = out.groupby("security")["turn"].diff(1)
    return output_factor(out, "wq_alpha_006", values)


def _make_wq_alpha_007(frame: pd.DataFrame) -> pd.DataFrame:
    out = sort_panel(frame)
    values = ((out["close"] - out["preclose"]) / out["preclose"].replace(0, pd.NA)) / out["turn"].replace(0, pd.NA)
    return output_factor(out, "wq_alpha_007", values)


def _make_wq_alpha_008(frame: pd.DataFrame) -> pd.DataFrame:
    out = sort_panel(frame)
    values = out.groupby("security")["close"].pct_change(10, fill_method=None) - out.groupby(
        "security"
    )["close"].pct_change(3, fill_method=None)
    return output_factor(out, "wq_alpha_008", values)


def _make_wq_alpha_009(frame: pd.DataFrame) -> pd.DataFrame:
    out = sort_panel(frame)
    values = ((out["high"] - out["close"]) - (out["close"] - out["low"])) / (
        (out["high"] - out["low"]).replace(0, pd.NA)
    )
    return output_factor(out, "wq_alpha_009", values)


def _make_wq_alpha_010(frame: pd.DataFrame) -> pd.DataFrame:
    out = sort_panel(frame)
    values = out.groupby("security")["amount"].rolling(20, min_periods=1).mean().reset_index(level=0, drop=True) / out["amount"].replace(0, pd.NA)
    return output_factor(out, "wq_alpha_010", values)


FACTOR_FUNCTIONS: dict[str, callable] = {
    "wq_alpha_001": _make_wq_alpha_001,
    "wq_alpha_002": _make_wq_alpha_002,
    "wq_alpha_003": _make_wq_alpha_003,
    "wq_alpha_004": _make_wq_alpha_004,
    "wq_alpha_005": factor_wq_alpha_005,
    "wq_alpha_006": _make_wq_alpha_006,
    "wq_alpha_007": _make_wq_alpha_007,
    "wq_alpha_008": _make_wq_alpha_008,
    "wq_alpha_009": _make_wq_alpha_009,
    "wq_alpha_010": _make_wq_alpha_010,
}
