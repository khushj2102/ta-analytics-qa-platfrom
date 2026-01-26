from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    exports_dir: Path
    cache_dir: Path
    manifest_path: Path
    events_path: Path
    errors_path: Path

def _next_seq(runs_dir: Path) -> int:
    runs_dir.mkdir(parents=True, exist_ok=True)
    seq = 0
    for p in runs_dir.iterdir():
        if p.is_dir() and "run-" in p.name:
            tail = p.name.split("run-")[-1]
            if tail.isdigit():
                seq = max(seq, int(tail))
    return seq + 1

def create_run(runs_dir: Path) -> RunContext:
    seq = _next_seq(runs_dir)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{ts}_run-{seq:06d}"
    run_dir = runs_dir / run_id
    exports_dir = run_dir / "exports"
    cache_dir = run_dir / "cache"
    run_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    return RunContext(
        run_id=run_id,
        run_dir=run_dir,
        exports_dir=exports_dir,
        cache_dir=cache_dir,
        manifest_path=run_dir / "run_manifest.json",
        events_path=run_dir / "events.jsonl",
        errors_path=run_dir / "errors.log",
    )

def write_manifest(ctx: RunContext, manifest: Dict[str, Any]) -> None:
    ctx.manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

def append_error(ctx: RunContext, text: str) -> None:
    with ctx.errors_path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")
