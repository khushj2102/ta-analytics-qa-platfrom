from __future__ import annotations
import pandas as pd
from typing import Tuple, List, Dict
from ..db.odbc import conncect_odbc, execute_sql_df
from .sql_templates import build_positions_flights_lookup_sql

def fetch_fr24_flight_ids(df_ref: pd.DataFrame, dsn_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    conn = conncect_odbc(dsn_name)
    results: List[pd.DataFrame] = []
    errors: List[Dict] = []
    
    try:
        for _,row in df_ref.iterrows():
            q_idx = row.get("query_index")
            reg = str(row.get("registration") or "")
            dep_iata = str(row.get("dep_iata") or "")
            arr_iata = str(row.get("arr_iata") or "")
            date_part = str(row.get("date_part_str") or "")

            if not(reg and dep_iata and arr_iata and date_part):
                errors.append({"query_index":q_idx, "error":"missing required fields", **row.to_dict()})
                continue

            sql = build_positions_flights_lookup_sql(reg, dep_iata, arr_iata, date_part)
            df1 = execute_sql_df(sql,conn)

            if df1.empty and "-" in reg:
                reg2 = reg.replace("-","")
                sql2 = build_positions_flights_lookup_sql(reg2, dep_iata, arr_iata, date_part)
                df1 = execute_sql_df(sql2, conn)
                if not df1.empty:
                    df1["match_type"] = "hyphen_removed"
            
            df1["query_index"] = q_idx
            results.append(df1)

    finally:
        try:
            conn.close()
        except Exception:
            pass
    results_df = pd.concat(results, ignore_index=True, sort=False) if results else pd.DataFrame()
    errors_df = pd.DataFrame(errors)
    return results_df, errors_df