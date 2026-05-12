# MFE5210 Alpha Factors Final Project

This repository reproduces an A-share alpha-factor research workflow for the MFE5210 final assignment. It builds a broad candidate factor library, applies production-style factor screening under a maximum absolute correlation threshold of 0.5, and backtests a market-neutral long-short strategy.

## Scope

- Download A-share daily data from Baostock and store it locally as parquet files.
- Build a research-ready panel from 2012-01-01 to 2026-04-20.
- Generate 60 candidate factors from momentum, reversal, volatility, liquidity, valuation, and WQ-style price-volume families.
- Evaluate each factor using Sharpe, long-short Sharpe, IC, Rank IC, IC IR, Rank IC IR, monthly win rate, and reversal direction.
- Select a final factor set with robust scoring, correlation clustering, and a final maximum pairwise absolute correlation constraint of 0.5.
- Backtest equal-weight, smoothed, and optimizer-based dynamic market-neutral long-short strategies and generate English and Chinese LaTeX reports.

## Data Source and Universe

The main data source is Baostock, because the trial JQData account available for this project does not cover the full 2012-2026 sample period. The primary universe is `hs300_zz500`, the union of CSI 300 and CSI 500 stocks used in the project panel. The Baostock industry file is a static snapshot used for practical cross-sectional controls, not a point-in-time historical industry database. The downloader also supports `all`, `sz50`, `hs300`, and `zz500`.

Raw data and generated factor stores are ignored by git to keep the repository lightweight. They can be reproduced with the commands below.

## Reproduction

Install the package in editable mode, then run the hardened report workflow:

```bash
python -m pip install -e ".[dev]"
PYTHONPATH=src python scripts/backup_report_state.py
PYTHONPATH=src python scripts/run_enhanced_model_backtest.py
PYTHONPATH=src python scripts/build_report_assets.py
PYTHONPATH=src python scripts/build_latex_reports.py
PYTHONPATH=src python scripts/package_submission.py
```

For PowerShell, set `PYTHONPATH` with `$env:PYTHONPATH='src'` before each command or once for the session. For a fast data smoke test, add `--limit 5` or another small number to `run_data_pipeline.py`.

## Project Structure

```text
src/mfe5210_alpha/data/       Data download and research panel construction
src/mfe5210_alpha/factors/    Candidate factor definitions
src/mfe5210_alpha/research/   Single-factor evaluation and correlation selection
src/mfe5210_alpha/model/      Composite score and long-short backtest
src/mfe5210_alpha/report/     Report tables, figures, and references
scripts/                      Reproducible command-line entry points
report/                       LaTeX source, PDF, figures, tables, and bibliography
report_zh/                    Chinese review report, kept outside submission
tests/                        Unit tests for the research pipeline
```

## References

The bibliography is generated into `report/references.bib`. It includes WorldQuant 101 Formulaic Alphas and the broker research reports used to motivate the daily momentum, reversal, volatility, turnover, liquidity, and price-volume factor families.
