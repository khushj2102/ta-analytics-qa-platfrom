from __future__ import annotations
import pandas as pd
import numpy as np

def add_heartbeat_trigger(
        df: pd.DataFrame,
        group_key: str = "flight_id",
        ts_col: str = "observation_timestamp",
        edr_col: str = "peak_edr",
        gap_minutes: float = 15.0,
        edr_threshold: float = 0.1
)-> pd.DataFrame:
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], format = "ISO8601", errors="coerce")
    df[edr_col] = df[edr_col].fillna(0.0) if edr_col in df.columns else 0.0

    df = df.sort_values([group_key, ts_col, "id"], ascending=True).reset_index(drop=True)
    df["time_gap_min"] = df.groupby(group_key)[ts_col].diff().dt.total_seconds().div(60).fillna(0.0)

    is_first = df.groupby(group_key).cumcount()==0
    gap_ge = df["time_gap_min"] >= gap_minutes
    edr_hi = df[edr_col]>edr_threshold
    gap_lt = ~gap_ge

    df["point_category"] = "heartbeat"
    df.loc[~is_first & gap_ge & edr_hi, "point_category"] = "trigger"
    df.loc[~is_first & gap_lt & edr_hi, "point_category"] = "trigger"

    df["non_trigger_low_gap_flag"] = np.where((~is_first) & gap_lt & (~edr_hi),1,0)

    return df