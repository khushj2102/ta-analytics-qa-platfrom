from __future__ import annotations

import streamlit as st
import pandas as pd
import traceback
from pathlib import Path
import json

from src.config import CONFIG
from src.db.sql_guard import validate_select_only, ensure_limit
from src.db.odbc import connect_odbc, execute_sql_df
from src.db.query_builder import TaQueryParams, build_ta_sql, params_to_dict
from src.logging_audit.run_manager import create_run, write_manifest, append_error
from src.logging_audit.event_logger import EventLogger
from src.enrich.candidate_key import ensure_candidate_key
from src.enrich.flight_id_ta import assign_flight_ids_ta
from src.enrich.profiling import add_profiling_columns
from src.enrich.heartbeat_trigger import add_heartbeat_trigger
from src.cache.session_cache import cache_get, cache_set, cache_clear
from src.cache.parquet_cache import save_parquet
from src.viz.map_plotly import build_map_figure
from src.viz.timeline_plotly import build_timeline
from src.export.export_manager import export_csv
from src.export.filename_builder import build_export_name

from src.fr24.airport_mapping import load_airport_mapping, map_icao_to_iata
from src.fr24.flight_lookup import fetch_fr24_flight_ids
from src.fr24.positions_fetch import fetch_fr24_positions

from dtale.app import get_instance
import dtale


st.set_page_config(page_title="TA Analytics & QA Master Platform", layout="wide")

def normalize_ta_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map common TA column variants -> required snake_case columns for enrichment.
    Required by assign_flight_ids_ta:
      - observation_timestamp, id, latitude, longitude
    """
    df = df.copy()

    # build a lowercase lookup of existing columns
    lower_map = {c.lower(): c for c in df.columns}

    def _rename_if_present(target: str, candidates: list[str]) -> None:
        for cand in candidates:
            if cand.lower() in lower_map and target not in df.columns:
                df.rename(columns={lower_map[cand.lower()]: target}, inplace=True)
                break

    # timestamps
    _rename_if_present("observation_timestamp", [
        "observation_timestamp",
        "observationtime", "observation_time", "observationdatetime",
        "timestamp", "time", "ts", "event_time",
        "observation_timestamp_utc", "observation_time_utc"
    ])

    # id
    _rename_if_present("id", [
        "id", "observation_id", "row_id", "point_id", "track_id"
    ])

    # lat/lon
    _rename_if_present("latitude", ["latitude", "lat", "y", "y_deg"])
    _rename_if_present("longitude", ["longitude", "lon", "lng", "long", "x", "x_deg"])

    # OPTIONAL: if altitude exists under different names, normalize it too
    _rename_if_present("altitude", ["altitude", "alt", "alt_ft", "altitude_ft", "altitude_feet"])

    return df


def build_run_manifest(
    ctx,
    *,
    tab: str,
    dsn: str,
    status: str,
    sql: str,
    params: dict,
    extra: dict | None = None,
) -> dict:
    """Standardize run_manifest.json structure and ensure audit/enrich defaults exist.

    - Puts audit output inside the run folder by default (run_dir/audit)
    - Provides default thresholds for flight_id_ta that can be overridden later
    """
    base = {
        "run_id": ctx.run_id,
        "tab": tab,
        "dsn": dsn,
        "status": status,
        "sql": sql,
        "params": params or {},
        "audit": {
            "enabled": True,
            "dir": str((ctx.run_dir / "audit").resolve()),
        },
        "enrich": {
            "flight_id_ta": {
                "enabled": True,
                "thresholds": {
                    "max_time_gap_s": 1800,
                    "max_distance_km": 150.0,
                    "max_speed_kmh": 1300.0,
                    "min_points_per_flight": 10,
                    "drop_small_segments": False,
                },
            }
        },
    }
    if extra:
        base.update(extra)
    return base





def run_query(dsn: str, sql: str, params_manifest: dict | None, tab: str) -> pd.DataFrame:
    ctx = create_run(CONFIG.runs_dir)
    ev = EventLogger(ctx.events_path)
    ev.log("run_created", tab=tab, run_id=ctx.run_id)

    st.session_state["last_run_dir"] = str(ctx.run_dir)
    st.session_state["last_run_id"] = ctx.run_id
    st.session_state["last_manifest_path"] = str(ctx.manifest_path)

    guard = validate_select_only(sql)
    if not guard.ok:
        write_manifest(ctx, build_run_manifest(ctx, tab=tab, dsn=dsn, status="blocked", sql=sql, params=params_manifest or {}, extra={"message": guard.message}))
        raise ValueError(guard.message)

    sql2, eff_limit, appended = ensure_limit(guard.sql, CONFIG.default_auto_limit, CONFIG.hard_max_rows)
    ev.log("sql_prepared", appended_limit=appended, effective_limit=eff_limit)

    try:
        ev.log("query_started", dsn=dsn)
        conn = connect_odbc(dsn)
        df = execute_sql_df(sql2, conn)
        conn.close()
        ev.log("query_finished", rows=len(df))
    except Exception as e:
        tb = traceback.format_exc()
        append_error(ctx, tb)
        write_manifest(ctx, build_run_manifest(ctx, tab=tab, dsn=dsn, status="error", sql=sql2, params=params_manifest or {}, extra={"error": str(e)}))
        raise

    if len(df) > CONFIG.hard_max_rows:
        df = df.iloc[: CONFIG.hard_max_rows].copy()

    write_manifest(ctx, build_run_manifest(ctx, tab=tab, dsn=dsn, status="ok", sql=sql2, params=params_manifest or {}, extra={
        "rows": int(len(df)),
        "columns": list(df.columns),
        "hard_max_rows": CONFIG.hard_max_rows,
        "default_auto_limit": CONFIG.default_auto_limit,
    }))

    try:
        save_parquet(df, ctx.cache_dir / f"{tab}_raw.parquet")
        ev.log("parquet_cached")
    except Exception:
        ev.log("parquet_cache_failed")

    st.session_state["last_run_dir"] = str(ctx.run_dir)
    return df

st.title("Turbulence Aware: Analytics & QA Master Platform")

with st.sidebar:
    st.header("Controls")
    if st.button("Clear session cache"):
        cache_clear()
        st.success("Session cache cleared.")

tabs = st.tabs(["Tab 1: TA Flight Info", "Tab 2: TA vs FR24", "Tab 3: TA EDA (D-Tale)"])

with tabs[0]:
    st.subheader("TA: Query, Enrich, Visualize, Export")

    mode = st.radio("Query mode", ["SQL", "Form"], horizontal=True)

    sql_text = ""
    params_manifest = {}
    if mode == "SQL":
        sql_text = st.text_area("SQL (SELECT-only)", height=200, placeholder="Paste a SELECT query here...")
    else:
        c1, c2 = st.columns(2)
        with c1:
            dt_start = st.text_input("DateTimeRange start (UTC ISO8601)", value="")
            dt_end = st.text_input("DateTimeRange end (UTC ISO8601)", value="")
            date_only = st.text_input("Date (YYYY-MM-DD) fallback", value="")
        with c2:
            airline = st.text_input("Airline", value="")
            callsign = st.text_input("Callsign", value="")
            reg = st.text_input("Registration", value="")
            dep = st.text_input("Departure (ICAO)", value="")
            arr = st.text_input("Arrival (ICAO)", value="")

        params = TaQueryParams(
            dt_start_utc=dt_start or None,
            dt_end_utc=dt_end or None,
            date_yyyy_mm_dd=date_only or None,
            airline=airline or None,
            callsign=callsign or None,
            registration=reg or None,
            departure_aerodome=dep or None,
            destination_aerodome=arr or None,
        )
        params_manifest = params_to_dict(params)
        sql_text = build_ta_sql(params)
        st.code(sql_text, language="sql")

    if st.button("Execute TA query", type="primary"):
        try:
            df = run_query(CONFIG.ta_dsn, sql_text, params_manifest=params_manifest, tab="tab1_ta")

            df = normalize_ta_columns(df)

            cache_set("tab1_raw", df)
            st.success(f"Loaded {len(df):,} rows from TA.")
        except Exception as e:
            st.error(str(e))
            if st.session_state.get("last_run_dir"):
                st.caption(f"Run folder: {st.session_state['last_run_dir']}")

    df_raw = cache_get("tab1_raw")
    if isinstance(df_raw, pd.DataFrame) and not df_raw.empty:
        st.divider()
        st.subheader("Enrichment")

        # Assumes query returns snake_case columns:
        df_en = ensure_candidate_key(df_raw, col_name="candidate_key")

        #Load manifest from the run folder (so thersholds + audit apply)

        manifest_path = st.session_state.get("last_manifest_path")
        manifest = {}
        if manifest_path and Path(manifest_path).exists():
            try:
                manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            except Exception:
                manifest = {}

        run_id = st.session_state.get("last_run_id")


        #Flight_id enrichment (TA segmentation)

        if "flight_id" not in df_en.columns:
            try:
                # This writes audit JSONL based on manifest["audit"] and thresholds under enrich.flight_id_ta.thresholds
                required = ["candidate_key", "observation_timestamp", "id", "latitude", "longitude"]
                missing = [c for c in required if c not in df_en.columns]
                if missing:
                    st.error(f"Missing columns for flight_id enrichment: {missing}")
                    st.caption(f"Available columns: {list(df_en.columns)}")
                    st.stop()
                df_en = assign_flight_ids_ta(df_en, manifest=manifest, run_id=run_id)

                # Log enrichment event in run events
                try:
                    # best-effort, don’t fail UI if logger path missing
                    run_dir = Path(st.session_state.get("last_run_dir", ""))
                    ev = EventLogger(run_dir / "events.jsonl")
                    ev.log("enrich_flight_id_ta_finished", rows=len(df_en), flight_ids=int(df_en["flight_id"].notna().sum()))
                except Exception:
                    pass

                st.success("flight_id assigned (TA segmentation).")
            except Exception as e:
                st.error(f"flight_id enrichment failed: {e}")
                st.stop()

        if "flight_id" in df_en.columns and df_en["flight_id"].notna().any():
            df_en = add_profiling_columns(df_en, group_key="flight_id")
            df_en = add_heartbeat_trigger(df_en, group_key="flight_id")
        else:
            st.warning("No non-null flight_id values produced (check thresholds / small segments).")

        cache_set("tab1_enriched", df_en)

        st.subheader("Column selector + sample")
        cols = list(df_en.columns)
        selected_cols = st.multiselect("Choose columns to display", options=cols, default=cols[: min(20, len(cols))])
        sort_cols = [c for c in ["observation_timestamp", "id"] if c in df_en.columns]
        sample = df_en.sort_values(sort_cols).head(10) if sort_cols else df_en.head(10)
        st.dataframe(sample[selected_cols] if selected_cols else sample, use_container_width=True)

        st.subheader("Flight selection")
        if "flight_id" in df_en.columns:
            flight_ids = sorted(df_en["flight_id"].dropna().unique().tolist())
            selected_flights = st.multiselect("Select flight_id(s) to plot", options=flight_ids, default=flight_ids[: min(3, len(flight_ids))])
            df_plot = df_en[df_en["flight_id"].isin(selected_flights)] if selected_flights else df_en.iloc[0:0]
        else:
            df_plot = df_en.iloc[0:0]

        # Visual sampling
        max_points = 200_000
        if len(df_plot) > max_points:
            df_vis = df_plot.sample(n=max_points, random_state=42)
            st.info(f"Visuals use a sampled subset of {max_points:,} rows (export remains full-fidelity).")
        else:
            df_vis = df_plot

        st.subheader("World map")
        show_points = st.checkbox("Show points", value=True)
        show_lines = st.checkbox("Show flight paths", value=True)

        must_hover = ["candidate_key", "latitude", "longitude", "altitude", "flight_id", "tafi", "id", "observation_index", "peak_edr"]
        hover_cols = [c for c in must_hover if c in df_vis.columns]

        if not df_vis.empty and all(c in df_vis.columns for c in ["latitude", "longitude"]):
            fig_map = build_map_figure(df_vis, lat_col="latitude", lon_col="longitude",
                                       color_col="flight_id" if "flight_id" in df_vis.columns else "candidate_key",
                                       hover_cols=hover_cols, show_points=show_points, show_lines=show_lines)
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.caption("Map will appear once latitude/longitude and flight_id exist.")

        st.subheader("Timeline")
        if not df_vis.empty and all(c in df_vis.columns for c in ["observation_timestamp", "altitude"]) and "speed_knots" in df_vis.columns:
            fig_t = build_timeline(df_vis, ts_col="observation_timestamp", altitude_col="altitude", speed_col="speed_knots")
            st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.caption("Timeline will appear once observation_timestamp/altitude/speed_knots exist.")

        st.subheader("Export")
        if st.button("Export full enriched CSV"):
            export_name = build_export_name(
                "ta_enriched",
                airline=df_en["airline"].iloc[0] if "airline" in df_en.columns and len(df_en) else None,
                registration=df_en["registration"].iloc[0] if "registration" in df_en.columns and len(df_en) else None,
                date_yyyy_mm_dd=str(df_en["observation_date"].iloc[0])[:10] if "observation_date" in df_en.columns and len(df_en) else None,
            )
            out_dir = Path(st.session_state.get("last_run_dir", "outputs")) / "exports"
            out_path = out_dir / export_name
            export_csv(df_en, out_path)
            st.success(f"Exported: {out_path}")

with tabs[1]:
    st.subheader("TA vs FR24 (visual overlay)")

    df_en = cache_get("tab1_enriched")
    if not isinstance(df_en, pd.DataFrame) or df_en.empty:
        st.info("Run Tab 1 first to generate enriched TA data with flight_id.")
    else:
        # Pick flights to compare
        if "flight_id" not in df_en.columns:
            st.warning("flight_id is missing in Tab 1 enriched data.")
        else:
            flight_ids = sorted(df_en["flight_id"].dropna().unique().tolist())
            if not flight_ids:
                st.warning("No flight_id values available to compare.")
            else:
                selected_flights = st.multiselect(
                    "Select TA flight_id(s) to fetch FR24 for",
                    options=flight_ids,
                    default=flight_ids[: min(3, len(flight_ids))],
                    key="tab2_flights",
                )
                if not selected_flights:
                    st.caption("Select at least one flight_id to continue.")
                else:
                    # Build reference rows for FR24 lookup
                    df_ref = (
                        df_en[df_en["flight_id"].isin(selected_flights)]
                        .groupby("flight_id", as_index=False)
                        .agg({
                            "observation_timestamp": "min",
                            "registration": "first" if "registration" in df_en.columns else "size",
                            "departure_aerodome": "first" if "departure_aerodome" in df_en.columns else "size",
                            "destination_aerodome": "first" if "destination_aerodome" in df_en.columns else "size",
                        })
                    )

                    # Clean up placeholders if missing columns existed
                    if "registration" not in df_en.columns:
                        df_ref["registration"] = ""
                    if "departure_aerodome" not in df_en.columns:
                        df_ref["departure_aerodome"] = ""
                    if "destination_aerodome" not in df_en.columns:
                        df_ref["destination_aerodome"] = ""

                    df_ref = df_ref.reset_index(drop=True)
                    df_ref["query_index"] = df_ref.index.astype(int)
                    df_ref["date_part_str"] = pd.to_datetime(df_ref["observation_timestamp"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")

                    # Map ICAO->IATA using your airport dataset
                    try:
                        mapping = load_airport_mapping(CONFIG.airport_mapping_csv)
                        df_ref = map_icao_to_iata(df_ref, mapping)
                    except Exception as e:
                        st.warning(f"Airport mapping failed (dep/arr IATA may be blank): {e}")
                        df_ref["dep_iata"] = df_ref.get("dep_iata", "")
                        df_ref["arr_iata"] = df_ref.get("arr_iata", "")

                    st.subheader("FR24 lookup inputs")
                    st.dataframe(df_ref[["flight_id", "registration", "departure_aerodome", "destination_aerodome", "dep_iata", "arr_iata", "date_part_str", "query_index"]], use_container_width=True)

                    col_a, col_b = st.columns(2)
                    with col_a:
                        do_fetch = st.button("Fetch FR24 (flight_ids + positions)", type="primary")
                    with col_b:
                        batch_size = st.number_input("FR24 positions batch size", min_value=10, max_value=500, value=100, step=10)

                    if do_fetch:
                        # Step 1: lookup positions_flights
                        with st.spinner("Looking up FR24 flight_ids..."):
                            df_ids_raw, df_ids_err = fetch_fr24_flight_ids(df_ref, CONFIG.fr24_dsn)

                        if not df_ids_raw.empty:
                            # Normalize column name
                            if "FLIGHT_ID" in df_ids_raw.columns and "fr24_flight_id" not in df_ids_raw.columns:
                                df_ids_raw = df_ids_raw.rename(columns={"FLIGHT_ID": "fr24_flight_id"})
                            st.success(f"FR24 flight_id matches: {len(df_ids_raw):,}")
                        else:
                            st.warning("No FR24 flight_id matches returned.")

                        if df_ids_err is not None and not df_ids_err.empty:
                            st.warning(f"FR24 lookup issues: {len(df_ids_err):,}")
                            st.dataframe(df_ids_err.head(50), use_container_width=True)

                        cache_set("tab2_fr24_flight_ids", df_ids_raw)

                        # Step 2: fetch positions
                        if df_ids_raw is not None and not df_ids_raw.empty and "fr24_flight_id" in df_ids_raw.columns:
                            # Ensure required columns exist for positions fetch
                            df_ids_raw = df_ids_raw.merge(
                                df_ref[["query_index", "date_part_str"]],
                                on="query_index",
                                how="left",
                            )

                            with st.spinner("Fetching FR24 positions..."):
                                df_pos, df_pos_err = fetch_fr24_positions(df_ids_raw[["fr24_flight_id", "date_part_str", "query_index"]], CONFIG.fr24_dsn, batch_size=int(batch_size))

                            cache_set("tab2_fr24_positions", df_pos)
                            if df_pos is not None and not df_pos.empty:
                                st.success(f"FR24 positions: {len(df_pos):,} rows")
                            else:
                                st.warning("No FR24 positions returned.")

                            if df_pos_err is not None and not df_pos_err.empty:
                                st.warning(f"FR24 positions fetch errors: {len(df_pos_err):,}")
                                st.dataframe(df_pos_err.head(50), use_container_width=True)

                    # Visual overlay (if we already have FR24 positions cached)
                    df_pos = cache_get("tab2_fr24_positions")
                    if isinstance(df_pos, pd.DataFrame) and not df_pos.empty:
                        st.divider()
                        st.subheader("Overlay map (TA vs FR24)")

                        # Build a combined plotting frame with common lat/lon and source label
                        ta_plot = df_en[df_en["flight_id"].isin(selected_flights)].copy()
                        ta_plot["source"] = "TA"
                        ta_plot = ta_plot.rename(columns={"latitude": "lat", "longitude": "lon"})
                        fr24_plot = df_pos.copy()
                        # best-effort normalize FR24 cols
                        if "fr24_latitude" in fr24_plot.columns:
                            fr24_plot = fr24_plot.rename(columns={"fr24_latitude": "lat", "fr24_longitude": "lon"})
                        fr24_plot["source"] = "FR24"

                        # keep only rows with lat/lon
                        plot_df = pd.concat(
                            [
                                ta_plot[[c for c in ["lat", "lon", "flight_id", "source", "observation_timestamp"] if c in ta_plot.columns]],
                                fr24_plot[[c for c in ["lat", "lon", "fr24_flight_id", "source", "fr24_obs_timestamp"] if c in fr24_plot.columns]].rename(columns={"fr24_flight_id": "flight_id", "fr24_obs_timestamp": "observation_timestamp"}),
                            ],
                            ignore_index=True,
                            sort=False,
                        ).dropna(subset=["lat", "lon"])

                        fig_map = build_map_figure(
                            plot_df.rename(columns={"lat": "latitude", "lon": "longitude"}),
                            lat_col="latitude",
                            lon_col="longitude",
                            color_col="source",
                            hover_cols=[c for c in ["source", "flight_id", "observation_timestamp"] if c in plot_df.columns],
                            show_points=True,
                            show_lines=True,
                        )
                        st.plotly_chart(fig_map, use_container_width=True)

with tabs[2]:
    st.subheader("TA EDA (D-Tale)")

    sql3 = st.text_area("SQL (SELECT-only)", height=200, key="tab3_sql")
    if st.button("Execute TA query (EDA)"):
        try:
            df3 = run_query(CONFIG.ta_dsn, sql3, params_manifest={}, tab="tab3_eda")
            df3 = normalize_ta_columns(df3)
            cache_set("tab3_raw", df3)
            st.success(f"Loaded {len(df3):,} rows.")

            # Start / reuse a D-Tale instance and provide a link
            try:
                inst = dtale.show(df3, open_browser=False, name=f"tab3_{st.session_state.get('last_run_id','run')}")
                url = inst._main_url  # e.g. http://localhost:40000/dtale/main/1
                st.session_state["tab3_dtale_url"] = url
            except Exception as e:
                st.warning(f"Could not start D-Tale: {e}")

        except Exception as e:
            st.error(str(e))
            if st.session_state.get("last_run_dir"):
                st.caption(f"Run folder: {st.session_state['last_run_dir']}")

    df3 = cache_get("tab3_raw")
    if isinstance(df3, pd.DataFrame) and not df3.empty:
        st.subheader("Preview")
        st.dataframe(df3.head(25), use_container_width=True)

        url = st.session_state.get("tab3_dtale_url")
        if url:
            st.markdown(f"✅ **D-Tale is running:** [Open D-Tale in a new tab]({url})")
        else:
            st.caption("Run the EDA query to launch D-Tale and get a link.")
