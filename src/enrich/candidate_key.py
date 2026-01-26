from __future__ import annotations

import pandas as pd

KEY_PARTS = ["airline","callsign", "registration", "departure_aerodome", "destination_aerodome"]

def ensure_candidate_key(df:pd.DataFrame, col_name: str = "cnadidate_key")-> pd.DataFrame:
    df = df.copy()
    if col_name in df.columns:
        return df
    
    for c in KEY_PARTS:
        if c not in df.columns:
            df[c] = ""
    
    def norm(x):
        if pd.isna(x):
            return ""
        return str(x).strip().upper()
    
    parts = [df[c].map(norm) for c in KEY_PARTS]
    out = parts[0].astype(str)
    for p in parts[1:]:
        out = out + "|" + p.astype(str)

    df[col_name] = out

    return df