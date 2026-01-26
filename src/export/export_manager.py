from __future__ import annotations
import pandas as pd
from pathlib import Path

def export_csv(df: pd.DataFrame, path: Path)-> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)