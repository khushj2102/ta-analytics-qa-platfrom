from __future__ import annotations
import pandas as pd
from typing import Tuple, List, Dict
from ..db.odbc import conncect_odbc, execute_sql_df
from .sql_templates import build_positions_sql

def fetch_fr24_positions(df_flight_ids: pd.DataFrame, dsn_name: str, batch_size: int = 100)-> Tuple[pd.DataFrame, pd.DataFrame]:
    required = {"fr24_flight_id","date_part_str","query_index"}
    missing = required - set(df_flight_ids.columns)
    if missing:
        raise ValueError(f"df_flight_ids missing columns: {missing}")
    
    conn = conncect_odbc(dsn_name)
    results: List[pd.DataFrame] = []
    errors: List[Dict] = []

    try:
        grouped = (
            df_flight_ids.dropna(subset=["fr24_flight_id","date_part_str"])
            .drop_duplicates(subset=["query_index","date_part_str","fr24_flight_id"])
            .groupby(["query_index","date_part_str"])["fr24_flight_id"].apply(list).reset_index()
        )

        for _, g in grouped.iterrows():
            q_idx = g["query_index"]
            date_part = str(g["date_part_str"])
            ids: List = g["fr24_flight_id"]
            chunks = [ids[i:i+batch_size] for i in range(0,len(ids), batch_size)]
            for batch_num , chunk in enumerate(chunks, start=1):
                try:
                    sql = build_positions_sql(chunk, date_part)
                    dfp = execute_sql_df(sql, conn)
                    dfp["query_index"] = q_idx
                    results.append(dfp)
                except Exception as e:
                    errors.append({
                        "query_index":q_idx,
                        "date_part_str": date_part,
                        "batch_num": batch_num,
                        "batch_size":len(chunk),
                        "error": f"{type(e).__name__}:{e}"
                    })
    finally:
        try:
            conn.close()
        except Exception:
            pass

        positions_df = pd.concat(results, ignore_index=True, sort=False) if results else pd.DataFrame()
        errors_df = pd.DataFrame(errors)
        return positions_df, errors_df