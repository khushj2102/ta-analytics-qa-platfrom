from __future__ import annotations
from datetime import datetime
from typing import Optional

def build_export_name(prefix: str,
                      airline: Optional[str] = None,
                      registration: Optional[str] = None,
                      date_yyyy_mm_dd: Optional[str] = None) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [prefix, ts]
    if airline: parts.append(str(airline))
    if registration: parts.append(str(registration))
    if date_yyyy_mm_dd: parts.append(str(date_yyyy_mm_dd))
    safe = "_".join([p.replace(" ", "").replace("/", "-") for p in parts if p])
    return safe + ".csv"