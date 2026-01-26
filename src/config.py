from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class AppConfig:
    ta_dsn: str = "ATHENA_MET_PROD"
    fr24_dsn: str = "ATHENA_FLIGHTRADAR_PROD"

    hard_max_rows: int = 1_000_000
    default_auto_limit: int = 100_000

    outputs_dir: Path = Path("outputs")
    runs_dir: Path = outputs_dir / "runs"

    airport_mapping_csv: Path = Path("data") / "airport_data_set_v1.csv"

CONFIG = AppConfig()
