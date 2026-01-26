from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

def build_timeline(df: pd.DataFrame,
                   ts_col: str = "observation_timestamp",
                   altitude_col: str = "altitude",
                   speed_col: str = "speed_knots",
                   extra_series: list[tuple[str,str]] | None = None) -> go.Figure:
    fig = go.Figure()
    if df is None or df.empty:
        return fig

    x = pd.to_datetime(df[ts_col], utc=True, errors="coerce")

    fig.add_trace(go.Scatter(x=x, y=df[altitude_col], name=altitude_col, yaxis="y1", mode="lines"))
    fig.add_trace(go.Scatter(x=x, y=df[speed_col], name=speed_col, yaxis="y2", mode="lines"))

    layout = dict(
        xaxis=dict(title="time (utc)", showspikes=True),
        yaxis=dict(title=altitude_col, showspikes=True),
        yaxis2=dict(title=speed_col, overlaying="y", side="right"),
        hovermode="x unified",
        margin=dict(l=40,r=40,t=30,b=40),
    )

    if extra_series:
        axis_count = 2
        for col, axis_title in extra_series:
            axis_count += 1
            ax = f"y{axis_count}"
            layout[ax] = dict(title=axis_title, overlaying="y", side="right", position=min(1.0, 0.98 - (axis_count-3)*0.05))
            fig.add_trace(go.Scatter(x=x, y=df[col], name=col, yaxis=ax, mode="lines"))

    fig.update_layout(**layout)
    return fig
