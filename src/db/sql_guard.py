from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Tuple

FORBIDDEN_KEYWORDS = ["INSERT", "UPDATE","DELETE","CREATE","DROP","ALTER","MERGE","TRUNCATE","GRANT","REVOKE","CALL","EXEC","EXECUTE"]

LIMIT_REGEX  = re.compile(r"\bLIMIT\s+(\d+)\b", re.IGNORECASE)

@dataclass(frozen=True)
class SqlGuardResult:
    ok: bool
    sql: str
    message: str

def validate_select_only(sql: str)-> SqlGuardResult:
    s = (sql or "").strip()
    if not s:
        return SqlGuardResult(False, sql, "SQL is empty.")
    
    if ";" in s[:-1]:
        return SqlGuardResult(False, sql, "Multiple statements detected. Only a single SELECT is allowed.")
    
    if not re.match(r"^\s*(SELECT|WITH)\b", s, flags=re.IGNORECASE):
        return SqlGuardResult(False, sql, "Only SELECT queries are allowed.")
    
    upper = s.upper()
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", upper):
            return SqlGuardResult(False, sql, f"Forbidden keyword detected: {kw}")

    s = re.sub(r";\s*$", "", s)
    return SqlGuardResult(True, s, "OK")

def ensure_limit(sql: str, default_limit: int, hard_cap: int) -> Tuple[str, int, bool]:
    s = re.sub(r";\s*$", "", (sql or "").strip())
    m = LIMIT_REGEX.search(s)
    if m:
        n = int(m.group(1))
        effective = min(n, hard_cap)
        if effective != n:
            s = LIMIT_REGEX.sub(f"LIMIT {effective}", s, count=1)
        return s, effective, False

    effective = min(default_limit, hard_cap)
    return f"{s}\nLIMIT {effective}", effective, True