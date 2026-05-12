from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mfe5210_matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _save_current_figure(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    return output_path


def plot_strategy_nav(daily_returns: pd.DataFrame, output_path: Path) -> Path:
    frame = daily_returns.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date")

    plt.figure(figsize=(7.2, 3.8))
    plt.plot(frame["date"], frame["cum_nav"], color="#1f5f5b", linewidth=1.8)
    plt.axhline(1.0, color="#999999", linewidth=0.8, linestyle="--")
    plt.title("Market-Neutral Long-Short Strategy NAV")
    plt.xlabel("Date")
    plt.ylabel("Cumulative NAV")
    plt.grid(alpha=0.25)
    return _save_current_figure(output_path)


def plot_strategy_drawdown(daily_returns: pd.DataFrame, output_path: Path) -> Path:
    frame = daily_returns.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date")

    plt.figure(figsize=(7.2, 3.4))
    plt.fill_between(frame["date"], frame["drawdown"], 0, color="#b44e3c", alpha=0.35)
    plt.title("Strategy Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(alpha=0.25)
    return _save_current_figure(output_path)


def plot_factor_sharpe_distribution(summary: pd.DataFrame, output_path: Path) -> Path:
    plt.figure(figsize=(7.2, 3.8))
    plt.hist(summary["ann_sharpe"].dropna(), bins=20, alpha=0.72, label="Factor return Sharpe")
    plt.hist(summary["ls_ann_sharpe"].dropna(), bins=20, alpha=0.52, label="Long-short Sharpe")
    plt.title("Single-Factor Sharpe Distribution")
    plt.xlabel("Annualized Sharpe")
    plt.ylabel("Factor count")
    plt.legend(frameon=False)
    plt.grid(alpha=0.20)
    return _save_current_figure(output_path)


def plot_selected_correlation(corr: pd.DataFrame, output_path: Path) -> Path:
    plt.figure(figsize=(6.4, 5.4))
    image = plt.imshow(corr, cmap="coolwarm", vmin=-0.5, vmax=0.5)
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.title("Selected Factor Correlation Matrix")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=6)
    return _save_current_figure(output_path)


def plot_selected_family_counts(selected: pd.DataFrame, output_path: Path) -> Path:
    counts = selected["family"].fillna("unknown").value_counts().sort_values(ascending=True)
    plt.figure(figsize=(6.6, 3.6))
    plt.barh(counts.index, counts.values, color="#315f72")
    plt.title("Selected Factors by Family")
    plt.xlabel("Factor count")
    plt.grid(axis="x", alpha=0.22)
    return _save_current_figure(output_path)
