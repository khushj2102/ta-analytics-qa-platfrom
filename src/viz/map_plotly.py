from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

def build_map_figure(df: pd.DataFrame,
                     lat_col: str = "latitude",
                     lon_col: str = "longitude",
                     color_col: str = "flight_id",
                     hover_cols: list[str] | None = None,
                     show_points: bool = True,
                     show_lines: bool = True) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,t=0,b=0))

    if df is None or df.empty:
        return fig

    hover_cols = hover_cols or []

    if show_points:
        fig.add_trace(go.Scattermapbox(
            lat=df[lat_col],
            lon=df[lon_col],
            mode="markers",
            marker=dict(size=6),
            text=["<br>".join([f"{c}: {row.get(c)}" for c in hover_cols]) for _, row in df.iterrows()],
            hoverinfo="text",
            name="points"
        ))

    if show_lines and color_col in df.columns:
        for fid, g in df.groupby(color_col):
            g = g.sort_values("observation_timestamp") if "observation_timestamp" in g.columns else g
            fig.add_trace(go.Scattermapbox(
                lat=g[lat_col],
                lon=g[lon_col],
                mode="lines",
                line=dict(width=2),
                name=str(fid)
            ))

    fig.update_layout(legend=dict(orientation="h"))
    return fig
