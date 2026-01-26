from __future__ import annotations
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

@dataclass
class EventLogger:
    events_path: Path

    def log(self, event: str, **fields: Any)->None:
        rec: Dict[str, Any] = {
            "ts":datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "event":event,
            **fields
        }
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")