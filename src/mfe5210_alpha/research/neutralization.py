from __future__ import annotations

import numpy as np
import pandas as pd


STYLE_CONTROL_COLUMNS = [
    "size_proxy",
    "liquidity_proxy",
    "volatility_proxy",
    "beta_proxy",
    "valuation_proxy",
]
EXPOSURE_STYLE_COLUMNS = ["log_adv20", "turnover_proxy", "member_hs300", "member_zz500"]
KEY_COLUMNS = ["date", "security"]


def _safe_zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    std = values.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (values - values.mean()) / std


def _fill_by_date(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column].replace([np.inf, -np.inf], np.nan)
    date_median = values.groupby(frame["date"]).transform(
        lambda x: x.median() if x.notna().any() else 0.0
    )
    filled = values.fillna(date_median)
    global_median = filled.median() if filled.notna().any() else 0.0
    return filled.fillna(global_median).fillna(0.0)


def _membership_series(panel: pd.DataFrame, *column_names: str) -> pd.Series:
    for column_name in column_names:
        if column_name in panel.columns:
            return pd.to_numeric(panel[column_name], errors="coerce").fillna(0.0).astype(float)
    return pd.Series(0.0, index=panel.index, dtype=float)


def _numeric_series(panel: pd.DataFrame, column_name: str, default: float = 0.0) -> pd.Series:
    if column_name in panel.columns:
        return pd.to_numeric(panel[column_name], errors="coerce")
    return pd.Series(default, index=panel.index, dtype=float)


def build_style_controls(panel: pd.DataFrame) -> pd.DataFrame:
    work = panel.sort_values(["security", "date"]).copy()
    turn = _numeric_series(work, "turn")
    turn_fraction = turn / 100.0
    amount = _numeric_series(work, "amount")
    float_mkt_proxy = amount.div(turn_fraction.where(turn_fraction > 0))

    work["size_proxy"] = np.log(float_mkt_proxy.where(float_mkt_proxy > 0))
    if "adv20" in work.columns:
        adv20 = pd.to_numeric(work["adv20"], errors="coerce")
    else:
        adv20 = (
            amount.groupby(work["security"])
            .rolling(20, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
    work["liquidity_proxy"] = np.log(adv20.where(adv20 > 0))

    ret = _numeric_series(work, "ret_1d")
    work["volatility_proxy"] = (
        ret.groupby(work["security"])
        .rolling(20, min_periods=5)
        .std()
        .reset_index(level=0, drop=True)
    )

    market_ret = ret.groupby(work["date"]).transform("mean")
    ret_market = ret * market_ret
    mean_ret = (
        ret.groupby(work["security"]).rolling(60, min_periods=20).mean().reset_index(level=0, drop=True)
    )
    mean_market = (
        market_ret.groupby(work["security"])
        .rolling(60, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )
    mean_ret_market = (
        ret_market.groupby(work["security"])
        .rolling(60, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )
    mean_market_sq = (
        (market_ret * market_ret)
        .groupby(work["security"])
        .rolling(60, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )
    cov = mean_ret_market - mean_ret * mean_market
    var = mean_market_sq - mean_market * mean_market
    work["beta_proxy"] = cov.div(var.where(var.abs() > 1e-12))

    valuation = _numeric_series(work, "pbMRQ")
    work["valuation_proxy"] = np.log(valuation.where(valuation > 0))

    controls = work[["date", "security", *STYLE_CONTROL_COLUMNS]].copy()
    for column in STYLE_CONTROL_COLUMNS:
        controls[column] = _fill_by_date(controls, column)
        controls[column] = controls.groupby("date")[column].transform(_safe_zscore)

    controls["log_adv20"] = np.log1p(adv20.clip(lower=0.0)).fillna(0.0).astype(float)
    controls["turnover_proxy"] = turn_fraction.fillna(0.0).astype(float)
    controls["member_hs300"] = _membership_series(work, "member_hs300", "hs300")
    controls["member_zz500"] = _membership_series(work, "member_zz500", "zz500")
    return controls


def neutralize_factor(
    frame: pd.DataFrame, control_cols: list[str], min_obs: int = 30
) -> pd.DataFrame:
    out_frames = []
    for _, group in frame.groupby("date", sort=True):
        result = group.copy()
        result["neutral_factor_value"] = np.nan
        complete = result[["factor_value", *control_cols]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(complete) < min_obs or not control_cols:
            result["neutral_factor_value"] = result["factor_value"]
            out_frames.append(result)
            continue

        y = pd.to_numeric(complete["factor_value"], errors="coerce").to_numpy(dtype=float)
        x_controls = complete[control_cols].apply(_safe_zscore).to_numpy(dtype=float)
        x = np.column_stack([np.ones(len(complete)), x_controls])
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        residual = y - x @ beta
        result.loc[complete.index, "neutral_factor_value"] = residual
        out_frames.append(result)
    return pd.concat(out_frames, ignore_index=True)


def neutralize_factor_matrix(
    frame: pd.DataFrame,
    factor_cols: list[str],
    control_cols: list[str],
    min_obs: int = 30,
) -> pd.DataFrame:
    out_frames = []
    neutral_cols = {column: f"{column}_neutral" for column in factor_cols}
    for _, group in frame.groupby("date", sort=True):
        result = group.copy()
        for neutral_col in neutral_cols.values():
            result[neutral_col] = np.nan

        if len(result) < min_obs or not control_cols:
            for factor_col, neutral_col in neutral_cols.items():
                result[neutral_col] = result[factor_col]
            out_frames.append(result)
            continue

        controls = result[control_cols].apply(pd.to_numeric, errors="coerce")
        controls = controls.replace([np.inf, -np.inf], np.nan)
        controls = controls.apply(_safe_zscore)
        valid_controls = controls.notna().all(axis=1).to_numpy()
        if valid_controls.sum() < min_obs:
            for factor_col, neutral_col in neutral_cols.items():
                result[neutral_col] = result[factor_col]
            out_frames.append(result)
            continue

        valid_index = result.index[valid_controls]
        x_controls = controls.loc[valid_index].to_numpy(dtype=float)
        x = np.column_stack([np.ones(len(valid_index)), x_controls])
        projection = np.linalg.pinv(x)

        factor_values = (
            result.loc[valid_index, factor_cols]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        observed = factor_values.notna()
        medians = factor_values.median(axis=0)
        filled_values = factor_values.fillna(medians).fillna(0.0).to_numpy(dtype=float)

        fitted = x @ (projection @ filled_values)
        residuals = filled_values - fitted
        residual_frame = pd.DataFrame(
            residuals,
            index=valid_index,
            columns=[neutral_cols[column] for column in factor_cols],
        )
        for factor_col, neutral_col in neutral_cols.items():
            residual_frame.loc[~observed[factor_col], neutral_col] = np.nan
        result.loc[valid_index, list(neutral_cols.values())] = residual_frame
        out_frames.append(result)
    return pd.concat(out_frames, ignore_index=True)


def _design_matrix(group: pd.DataFrame, use_industry: bool) -> pd.DataFrame:
    columns = [column for column in EXPOSURE_STYLE_COLUMNS if column in group.columns]
    design = group.loc[:, columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    design.insert(0, "const", 1.0)

    if use_industry and "industry" in group.columns:
        industry = group["industry"].fillna("missing").astype(str)
        dummies = pd.get_dummies(industry, prefix="industry", dtype=float)
        design = pd.concat([design, dummies], axis=1)

    return design.astype(float)


def _residualize_exposure(y: pd.Series, x: pd.DataFrame, method: str, weights: pd.Series) -> pd.Series:
    valid = y.notna()
    residuals = pd.Series(np.nan, index=y.index, dtype=float)
    if not valid.any():
        return residuals

    y_values = y.loc[valid].astype(float).to_numpy()
    x_values = x.loc[valid].to_numpy(dtype=float)

    if method == "wls":
        root_weights = np.sqrt(weights.loc[valid].to_numpy(dtype=float))
        y_values = y_values * root_weights
        x_values = x_values * root_weights[:, None]

    fitted = x.loc[valid].to_numpy(dtype=float) @ np.linalg.lstsq(x_values, y_values, rcond=None)[0]
    residuals.loc[valid] = y.loc[valid].astype(float).to_numpy() - fitted
    return residuals


def _neutralize_date(
    group: pd.DataFrame,
    factor_names: list[str],
    use_industry: bool,
    method: str,
) -> pd.DataFrame:
    out = group.loc[:, KEY_COLUMNS + factor_names].copy()
    design = _design_matrix(group, use_industry)
    weights = np.exp(pd.to_numeric(group["log_adv20"], errors="coerce").fillna(0.0))
    weights = pd.Series(np.maximum(weights, 1.0), index=group.index)

    for factor_name in factor_names:
        out[factor_name] = _residualize_exposure(group[factor_name], design, method, weights)
    return out


def neutralize_exposures(
    exposures: pd.DataFrame,
    controls: pd.DataFrame,
    factor_names: list[str],
    use_industry: bool = True,
    method: str = "ols",
) -> pd.DataFrame:
    if method not in {"ols", "wls"}:
        raise ValueError("method must be 'ols' or 'wls'")

    merged = exposures.merge(controls, on=KEY_COLUMNS, how="left")
    for column in EXPOSURE_STYLE_COLUMNS:
        if column not in merged.columns:
            merged[column] = 0.0
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)

    pieces = [
        _neutralize_date(group, list(factor_names), use_industry, method)
        for _, group in merged.groupby("date", sort=False)
    ]
    if not pieces:
        return exposures.loc[:, KEY_COLUMNS + list(factor_names)].copy()
    return pd.concat(pieces, ignore_index=True)
