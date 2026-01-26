from __future__ import annotations
import pyodbc
import pandas as pd
from typing import Optional

def connect_odbc(dsn_name: str)-> pyodbc.Connection:
    return pyodbc.connect(f"DSN={dsn_name};", autocommit=True)

def execute_sql_df(sql: str, conn: pyodbc.Connection, cursor_timeout: Optional[int] = None)-> pd.DataFrame:
    try:
        return pd.read_sql_query(sql, conn)
    except Exception:
        cur = conn.cursor()
        if cursor_timeout is not None:
            try:
                cur.timeout = cursor_timeout
            except Exception:
                pass
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [c[0] for c in cur.description] if cur.description else []
        return pd.DataFrame.from_records(rows, column=cols)