# MFE5210 Alpha Factors Final Submission

This submission keeps only the core research and backtest code, the final report PDF, and the reference papers used in the project.

## Included Files

- `src/mfe5210_alpha/`: data pipeline, factor construction, factor screening, backtest, and out-of-sample experiment code.
- `scripts/`: command-line entry points for reproducing the main workflow.
- `report/main.pdf`: final report submitted for the project.
- `references/`: source literature PDFs used to motivate the factor design.

## Data and Outputs

Raw market data, intermediate parquet stores, and generated backtest outputs are not included in this repository.

## Main Reproduction Flow

Install the package first:

```bash
python -m pip install -e .
```

Then run the main workflow:

```bash
python scripts/run_data_pipeline.py
python scripts/build_research_store.py
python scripts/run_factor_pipeline.py
python scripts/run_enhanced_factor_research.py
python scripts/select_enhanced_factors.py
python scripts/run_enhanced_model_backtest.py
python scripts/run_fixed_split_experiment.py
python scripts/run_walk_forward_experiment.py
python scripts/run_factor_fusion_grid.py --route all
python scripts/build_oos_comparison.py
```

In PowerShell, set `PYTHONPATH` with `$env:PYTHONPATH='src'` before running the scripts.
