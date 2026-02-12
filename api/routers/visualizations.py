"""
Visualization endpoints: return Plotly figures as JSON dicts.
"""

import json
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
import numpy as np

from api.dependencies import require_data
from api.session_store import SessionData
from api.models.requests import TimeSeriesRequest, GeospatialRequest
from api.models.responses import GeospatialResponse
from core.statistics import correlation_matrix
from core.visualizations import (
    create_correlation_heatmap,
    create_distribution_plot,
    create_scatter_plot,
    create_box_plot_by_group,
    create_count_plot,
    create_time_series_plot,
)
from core.geospatial import (
    detect_coordinate_candidates,
    validate_coordinates,
    create_geospatial_figure,
)

router = APIRouter(prefix="/api/viz", tags=["visualizations"])


def _fig_response(fig) -> JSONResponse:
    """Serialize a Plotly figure to a JSON response safely."""
    return JSONResponse(content={"figure": json.loads(fig.to_json())})


@router.get("/correlation-heatmap")
def correlation_heatmap(session: SessionData = Depends(require_data)):
    """Return Plotly correlation heatmap as JSON."""
    df = session.current_data
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 numeric columns")
    corr = correlation_matrix(df, num_cols)
    fig = create_correlation_heatmap(corr)
    return _fig_response(fig)


@router.get("/distribution/{col}")
def distribution(
    col: str,
    plot_type: str = Query("histogram", pattern="^(histogram|box|violin)$"),
    session: SessionData = Depends(require_data),
):
    """Distribution plot for a column."""
    df = session.current_data
    if col not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{col}' not found")
    fig = create_distribution_plot(df, col, plot_type=plot_type)
    return _fig_response(fig)


@router.get("/scatter/{x}/{y}")
def scatter(
    x: str,
    y: str,
    color: Optional[str] = None,
    trendline: bool = False,
    session: SessionData = Depends(require_data),
):
    """Scatter plot between two columns."""
    df = session.current_data
    for c in [x, y]:
        if c not in df.columns:
            raise HTTPException(status_code=404, detail=f"Column '{c}' not found")
    fig = create_scatter_plot(df, x, y, color_col=color, trendline=trendline)
    return _fig_response(fig)


@router.get("/boxplot/{numeric}/{group}")
def boxplot(
    numeric: str,
    group: str,
    session: SessionData = Depends(require_data),
):
    """Grouped box plot."""
    df = session.current_data
    for c in [numeric, group]:
        if c not in df.columns:
            raise HTTPException(status_code=404, detail=f"Column '{c}' not found")
    fig = create_box_plot_by_group(df, numeric, group)
    return _fig_response(fig)


@router.get("/countplot/{col}")
def countplot(
    col: str,
    top_n: int = 10,
    session: SessionData = Depends(require_data),
):
    """Count plot for a categorical column."""
    df = session.current_data
    if col not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{col}' not found")
    fig = create_count_plot(df, col, top_n=top_n)
    return _fig_response(fig)


@router.post("/time-series")
def time_series(
    body: TimeSeriesRequest,
    session: SessionData = Depends(require_data),
):
    """Time-series visualization for selected metrics."""
    df = session.current_data
    if not body.value_columns:
        raise HTTPException(status_code=400, detail="Select at least one value column")

    missing = [col for col in body.value_columns if col not in df.columns]
    if missing:
        raise HTTPException(status_code=404, detail=f"Columns not found: {', '.join(missing)}")

    x_col = body.x_column
    working_df = df.copy()
    if x_col:
        if x_col not in df.columns:
            raise HTTPException(status_code=404, detail=f"Column '{x_col}' not found")
        working_df[x_col] = pd.to_datetime(working_df[x_col], errors="coerce")
        working_df = working_df.dropna(subset=[x_col])
    else:
        x_col = "__sequence__"
        working_df = working_df.reset_index().rename(columns={"index": x_col})

    fig = create_time_series_plot(
        working_df,
        date_col=x_col,
        value_cols=body.value_columns,
        chart_type=body.chart_type,
        rolling_window=body.rolling_window,
    )
    return _fig_response(fig)


@router.post("/geospatial", response_model=GeospatialResponse)
def geospatial(
    body: GeospatialRequest,
    session: SessionData = Depends(require_data),
):
    """Render an interactive map after validating coordinate columns."""
    df = session.current_data
    candidates = detect_coordinate_candidates(df)

    lat_col = body.lat_column or (candidates[0].lat_column if candidates else None)
    lon_col = body.lon_column or (candidates[0].lon_column if candidates else None)

    if not lat_col or not lon_col:
        raise HTTPException(status_code=400, detail="No latitude/longitude columns detected")

    for col in [lat_col, lon_col]:
        if col not in df.columns:
            raise HTTPException(status_code=404, detail=f"Column '{col}' not found")

    color_col = body.color_column
    if color_col and color_col not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{color_col}' not found")
    if color_col is None:
        color_col = _default_color_column(df, {lat_col, lon_col})

    fig = create_geospatial_figure(df, lat_col, lon_col, color_col)
    validation = validate_coordinates(df, lat_col, lon_col)
    warnings = []
    if validation["valid_ratio"] < 0.5:
        warnings.append("Less than half of the records contain valid coordinates")

    candidate_payload = [
        {
            "lat_column": c.lat_column,
            "lon_column": c.lon_column,
            "lat_valid_ratio": c.lat_valid_ratio,
            "lon_valid_ratio": c.lon_valid_ratio,
        }
        for c in candidates
    ]

    return GeospatialResponse(
        figure=json.loads(fig.to_json()),
        lat_column=lat_col,
        lon_column=lon_col,
        color_column=color_col,
        validation=validation,
        warnings=warnings,
        candidates=candidate_payload,
    )


def _default_color_column(df: pd.DataFrame, exclude: set[str]) -> Optional[str]:
    for col in df.columns:
        if col in exclude:
            continue
        if df[col].dtype == object or getattr(df[col].dtype, "name", "") == "category":
            return col
    return None
