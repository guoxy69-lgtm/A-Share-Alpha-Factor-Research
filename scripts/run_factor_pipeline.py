from pathlib import Path

import pandas as pd

from mfe5210_alpha.config import ProjectConfig
from mfe5210_alpha.factors.catalog import build_factor_catalog
from mfe5210_alpha.factors.liquidity import FACTOR_FUNCTIONS as LIQUIDITY_FUNCTIONS
from mfe5210_alpha.factors.momentum import FACTOR_FUNCTIONS as MOMENTUM_FUNCTIONS
from mfe5210_alpha.factors.valuation import FACTOR_FUNCTIONS as VALUATION_FUNCTIONS
from mfe5210_alpha.factors.volatility import FACTOR_FUNCTIONS as VOLATILITY_FUNCTIONS
from mfe5210_alpha.factors.wq_like import FACTOR_FUNCTIONS as WQ_FUNCTIONS

FACTOR_FUNCTIONS = {
    **MOMENTUM_FUNCTIONS,
    **VOLATILITY_FUNCTIONS,
    **LIQUIDITY_FUNCTIONS,
    **VALUATION_FUNCTIONS,
    **WQ_FUNCTIONS,
}


if __name__ == "__main__":
    cfg = ProjectConfig.from_root(Path(__file__).resolve().parents[1])
    panel = pd.read_parquet(cfg.processed_root / "research_panel.parquet")

    metadata_rows = []
    for spec in build_factor_catalog():
        factor_frame = FACTOR_FUNCTIONS[spec.name](panel)
        path = cfg.factor_root / f"{spec.name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        factor_frame.to_parquet(path, index=False)
        metadata_rows.append(
            {
                "factor_name": spec.name,
                "family": spec.family,
                "description": spec.description,
                "formula_text": spec.formula_text,
                "source_quote": spec.source_quote,
                "file_path": str(path),
            }
        )

    metadata = pd.DataFrame(metadata_rows)
    metadata_path = cfg.factor_root / "factor_catalog.csv"
    metadata.to_csv(metadata_path, index=False)
    print(metadata_path)
