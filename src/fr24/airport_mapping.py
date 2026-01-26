from __future__ import annotations
import pandas as pd
from pathlib import Path

def load_airport_mapping(csv_path: Path)-> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = {"CD_LOCATIONICAO", "CD_LOCATIONIATA"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Airport mapping csv missing columns: {missing}")
    return df[["CD_LOCATIONICAO", "CD_LOCATIONIATA"]].drop_duplicates()

def map_icao_to_iata(df: pd.DataFrame, mapping: pd.DataFrame)-> pd.DataFrame:
    out = df.copy()
    out = out.merge(mapping, left_on="departure_aerodome", right_on="CD_LOCATIONICAO", how="left")
    out = out.rename(columns={"CD_LOCATIONIATA": "dep_iata"}).drop(columns=["CD_LOCATIONICAO"])
    out = out.merge(mapping, left_on="destination_aerodome", right_on="CD_LOCATIONICAO", how="left")
    out = out.rename(columns={"CD_LOCATIONIATA": "arr_iata"}).drop(columns=["CD_LOCATIONICAO"])
    return out