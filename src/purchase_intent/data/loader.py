from pathlib import Path

import pandas as pd


def load_raw_data(path: Path) -> pd.DataFrame:
    """Carrega o dataset bruto de compras a partir de um arquivo CSV."""
    return pd.read_csv(path)
