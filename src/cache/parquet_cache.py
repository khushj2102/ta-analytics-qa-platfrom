from __future__ import annotations

import pandas as pd
from pathlib import Path

def save_parquet(df: pd.DataFrame, path: Path)-> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def load_parquet(path: Path)-> pd.DataFrame:
    return pd.read_parquet(path)