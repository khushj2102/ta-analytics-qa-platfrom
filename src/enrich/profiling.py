from __future__ import annotations

import numpy as np
import pandas as pd

R_KM = 6371.0088
KM_TO_NM = 0.539957

def _haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R_KM * c

def add_profiling_columns(df: pd.DataFrame, group_key: str = "flight_id") -> pd.DataFrame:
    df = df.copy()
    df["observation_timestamp"] = pd.to_datetime(df["observation_timestamp"], utc=True, errors="coerce")
    df = df.sort_values([group_key, "observation_timestamp", "id"], ascending=True).reset_index(drop=True)

    df["no_of_observations"] = df.groupby(group_key)["id"].transform("nunique")
    df["observation_index"] = df.groupby(group_key).cumcount() + 1

    fields = ["id", "observation_timestamp", "latitude", "longitude", "altitude"]
    for f in fields:
        df[f"previous_{f}"] = df.groupby(group_key)[f].shift(1)
        df[f"next_{f}"] = df.groupby(group_key)[f].shift(-1)

    df["distance_km"] = _haversine_km(
        df["latitude"], df["longitude"],
        df["previous_latitude"], df["previous_longitude"]
    )
    df.loc[df["previous_latitude"].isna() | df["previous_longitude"].isna(), "distance_km"] = np.nan
    df["distance_nm"] = df["distance_km"] * KM_TO_NM

    df["time_diff_sec"] = (df["observation_timestamp"] - df["previous_observation_timestamp"]).dt.total_seconds()

    hours = df["time_diff_sec"] / 3600.0
    df["speed_kmph"] = np.where(hours > 0, df["distance_km"] / hours, 0.0)
    df["speed_knots"] = np.where(hours > 0, df["distance_nm"] / hours, 0.0)

    df["altitude_change_ft"] = df["altitude"] - df["previous_altitude"]
    df["altitude_change_m"] = df["altitude_change_ft"] * 0.3048

    minutes = df["time_diff_sec"] / 60.0
    df["vertical_rate_fpm"] = np.where(minutes > 0, df["altitude_change_ft"] / minutes, 0.0)

    return df