from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)

# A fixed namespace makes uuid5 deterministic across runs.
# Feel free to replace with your org/project UUID, but KEEP IT STABLE once deployed.
TA_FLIGHTID_NAMESPACE = uuid.UUID("2f9b0a2a-7f0d-4f3a-bb1a-9f2cc9f927f5")


@dataclass(frozen=True)
class TAFlightIdThresholds:
    max_time_gap_s: int = 30 * 60          # 30 minutes
    max_distance_km: float = 150.0         # large jump => new segment
    max_speed_kmh: float = 1300.0          # sanity check (jet upper bound-ish)
    min_points_per_flight: int = 10
    drop_small_segments: bool = False      # if True, flight_id becomes null for small segments


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        LOGGER.exception("Failed to parse JSON at %s", str(path))
        return {}


def _load_manifest(manifest: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Priority:
      1) manifest passed in
      2) RUN_MANIFEST_PATH env var
      3) ./run_manifest.json (cwd)
      4) {}
    """
    if manifest is not None:
        return manifest

    env_path = os.getenv("RUN_MANIFEST_PATH")
    if env_path:
        return _safe_read_json(Path(env_path))

    cwd_path = Path("run_manifest.json")
    if cwd_path.exists():
        return _safe_read_json(cwd_path)

    return {}


def _thresholds_from_manifest(manifest: Dict[str, Any]) -> TAFlightIdThresholds:
    cfg = (
        manifest.get("enrich", {})
        .get("flight_id_ta", {})
        .get("thresholds", {})
    )

    # Merge with defaults carefully.
    defaults = TAFlightIdThresholds()
    return TAFlightIdThresholds(
        max_time_gap_s=int(cfg.get("max_time_gap_s", defaults.max_time_gap_s)),
        max_distance_km=float(cfg.get("max_distance_km", defaults.max_distance_km)),
        max_speed_kmh=float(cfg.get("max_speed_kmh", defaults.max_speed_kmh)),
        min_points_per_flight=int(cfg.get("min_points_per_flight", defaults.min_points_per_flight)),
        drop_small_segments=bool(cfg.get("drop_small_segments", defaults.drop_small_segments)),
    )


def _audit_cfg_from_manifest(manifest: Dict[str, Any]) -> Tuple[bool, Path]:
    audit_cfg = manifest.get("audit", {}) or {}
    enabled = bool(audit_cfg.get("enabled", True))
    out_dir = Path(audit_cfg.get("dir", "artifacts/audit"))
    return enabled, out_dir


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    """
    Vector-friendly enough for apply; used row-wise here for clarity.
    """
    import math

    # Handle missing values
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return float("nan")

    r = 6371.0  # Earth radius (km)
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def _deterministic_flight_uuid(candidate_key: str, seg_idx: int, seg_start_iso: str) -> str:
    """
    Deterministic uuid5 based on stable inputs.
    seg_start_iso should be ISO string in UTC (or at least stable format).
    """
    name = f"ta|{candidate_key}|seg={seg_idx}|start={seg_start_iso}"
    return str(uuid.uuid5(TA_FLIGHTID_NAMESPACE, name))


def _write_audit_jsonl(
    out_dir: Path,
    record: Dict[str, Any],
    filename: str = "flight_id_ta_audit.jsonl",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def assign_flight_ids_ta(
    df_raw: pd.DataFrame,
    *,
    manifest: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    TA-only flight_id assignment.

    Requirements (input columns expected):
      - candidate_key
      - observation_timestamp
      - id
      - latitude, longitude
      - altitude (optional for segmentation, kept for context)
      - tafi (kept for context)

    Output:
      - flight_id (deterministic UUID string or NA if drop_small_segments=True and too small)
      - diagnostics columns:
          - flight_seg_idx
          - flight_seg_break_reason
          - flight_time_gap_s
          - flight_distance_km
          - flight_speed_kmh
          - flight_seg_start_ts
          - flight_seg_end_ts
          - flight_seg_points
    """
    manifest_obj = _load_manifest(manifest)
    thr = _thresholds_from_manifest(manifest_obj)
    audit_enabled, audit_dir = _audit_cfg_from_manifest(manifest_obj)

    if run_id is None:
        run_id = manifest_obj.get("run_id") or f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    required = ["candidate_key", "observation_timestamp", "id", "latitude", "longitude"]
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise ValueError(f"assign_flight_ids_ta missing required columns: {missing}")

    df = df_raw.copy()

    # Normalize timestamp
    df["observation_timestamp"] = pd.to_datetime(df["observation_timestamp"], utc=True, errors="coerce")
    if df["observation_timestamp"].isna().any():
        LOGGER.warning("Some observation_timestamp values could not be parsed (NaT). They will force new segments.")

    # Sort for deterministic segmentation
    df = df.sort_values(["candidate_key", "observation_timestamp", "id"], kind="mergesort").reset_index(drop=True)

    # Compute per-candidate previous values
    df["_prev_ts"] = df.groupby("candidate_key")["observation_timestamp"].shift(1)
    df["_prev_lat"] = df.groupby("candidate_key")["latitude"].shift(1)
    df["_prev_lon"] = df.groupby("candidate_key")["longitude"].shift(1)

    # Time gap (seconds)
    df["flight_time_gap_s"] = (df["observation_timestamp"] - df["_prev_ts"]).dt.total_seconds()

    # Distance km
    df["flight_distance_km"] = df.apply(
        lambda r: _haversine_km(r["_prev_lat"], r["_prev_lon"], r["latitude"], r["longitude"]),
        axis=1,
    )

    # Speed km/h
    def _speed_kmh(time_gap_s: Any, dist_km: Any) -> float:
        if pd.isna(time_gap_s) or pd.isna(dist_km) or time_gap_s <= 0:
            return float("nan")
        return float(dist_km) / (float(time_gap_s) / 3600.0)

    df["flight_speed_kmh"] = df.apply(
        lambda r: _speed_kmh(r["flight_time_gap_s"], r["flight_distance_km"]),
        axis=1,
    )

    # Determine segment breaks
    break_reason = []

    for i, r in df.iterrows():
        if pd.isna(r["_prev_ts"]) or pd.isna(r["observation_timestamp"]):
            break_reason.append("start_or_bad_ts")
            continue

        # Missing coordinates => break
        if pd.isna(r["_prev_lat"]) or pd.isna(r["_prev_lon"]) or pd.isna(r["latitude"]) or pd.isna(r["longitude"]):
            break_reason.append("missing_coord")
            continue

        tg = r["flight_time_gap_s"]
        dk = r["flight_distance_km"]
        sp = r["flight_speed_kmh"]

        if pd.isna(tg) or tg > thr.max_time_gap_s:
            break_reason.append("time_gap")
        elif (not pd.isna(dk)) and dk > thr.max_distance_km:
            break_reason.append("distance_jump")
        elif (not pd.isna(sp)) and sp > thr.max_speed_kmh:
            break_reason.append("speed_jump")
        else:
            break_reason.append("")

    df["flight_seg_break_reason"] = break_reason

    # Segment index per candidate_key
    # New segment whenever break_reason != "" OR first row in group.
    is_new_seg = (df["flight_seg_break_reason"] != "") | df["_prev_ts"].isna()
    # pandas-safe: cumsum a real column (not a boolean mask)
    df["_is_new_seg"] = is_new_seg.astype(int)
    df["flight_seg_idx"] = df.groupby("candidate_key")["_is_new_seg"].cumsum().astype(int) - 1
    df.loc[df["flight_seg_idx"] < 0, "flight_seg_idx"] = 0

    # Segment start/end per (candidate_key, seg_idx)
    seg_g = df.groupby(["candidate_key", "flight_seg_idx"], dropna=False)

    df["flight_seg_start_ts"] = seg_g["observation_timestamp"].transform("min")
    df["flight_seg_end_ts"] = seg_g["observation_timestamp"].transform("max")
    df["flight_seg_points"] = seg_g["id"].transform("count").astype(int)

    # Build deterministic flight_id
    # Use seg_start_ts as stable anchor
    df["_seg_start_iso"] = df["flight_seg_start_ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df["flight_id"] = df.apply(
        lambda r: _deterministic_flight_uuid(str(r["candidate_key"]), int(r["flight_seg_idx"]), str(r["_seg_start_iso"])),
        axis=1,
    )

    # Optionally drop too-small segments
    if thr.drop_small_segments:
        too_small = df["flight_seg_points"] < thr.min_points_per_flight
        df.loc[too_small, "flight_id"] = pd.NA
        # keep reason in audit; do not overwrite break_reason

    # Audit summary
    if audit_enabled:
        reason_counts = (
            df["flight_seg_break_reason"]
            .replace("", "no_break")
            .value_counts(dropna=False)
            .to_dict()
        )
        seg_sizes = (
            df.groupby(["candidate_key", "flight_seg_idx"])["id"]
            .count()
            .describe()
            .to_dict()
        )

        audit_record = {
            "ts_utc": _utc_now_iso(),
            "run_id": run_id,
            "component": "enrich.flight_id_ta",
            "thresholds": {
                "max_time_gap_s": thr.max_time_gap_s,
                "max_distance_km": thr.max_distance_km,
                "max_speed_kmh": thr.max_speed_kmh,
                "min_points_per_flight": thr.min_points_per_flight,
                "drop_small_segments": thr.drop_small_segments,
            },
            "input_rows": int(len(df_raw)),
            "output_rows": int(len(df)),
            "candidate_keys": int(df["candidate_key"].nunique(dropna=True)),
            "segments": int(df[["candidate_key", "flight_seg_idx"]].drop_duplicates().shape[0]),
            "flight_ids_non_null": int(df["flight_id"].notna().sum()),
            "break_reason_counts": reason_counts,
            "segment_size_summary": seg_sizes,
            "time_range_utc": {
                "min_ts": None if df["observation_timestamp"].isna().all() else df["observation_timestamp"].min().isoformat(),
                "max_ts": None if df["observation_timestamp"].isna().all() else df["observation_timestamp"].max().isoformat(),
            },
        }

        _write_audit_jsonl(audit_dir, audit_record)

    # Cleanup internal columns
    df = df.drop(
    columns=["_prev_ts", "_prev_lat", "_prev_lon", "_seg_start_iso", "_is_new_seg"],
    errors="ignore",
    )

    return df
