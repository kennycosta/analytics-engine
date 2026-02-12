"""Geospatial helpers for detecting latitude/longitude columns and building maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


@dataclass
class CoordinateCandidate:
    lat_column: str
    lon_column: str
    lat_valid_ratio: float
    lon_valid_ratio: float


COORDINATE_THRESHOLD = 0.6


def detect_coordinate_candidates(df: pd.DataFrame) -> List[CoordinateCandidate]:
    candidates: List[CoordinateCandidate] = []
    if df is None or df.empty:
        return candidates

    lat_scores = {
        col: _valid_ratio(df[col], -90, 90)
        for col in df.columns
    }
    lon_scores = {
        col: _valid_ratio(df[col], -180, 180)
        for col in df.columns
    }

    common_pairs = [
        ("latitude", "longitude"),
        ("lat", "lon"),
        ("lat", "lng"),
    ]

    lower_map = {col.lower(): col for col in df.columns}
    for lat_key, lon_key in common_pairs:
        if lat_key in lower_map and lon_key in lower_map:
            lat_col = lower_map[lat_key]
            lon_col = lower_map[lon_key]
            candidates.append(
                CoordinateCandidate(
                    lat_column=lat_col,
                    lon_column=lon_col,
                    lat_valid_ratio=lat_scores.get(lat_col, 0.0),
                    lon_valid_ratio=lon_scores.get(lon_col, 0.0),
                )
            )

    # Pair top scoring columns if explicit names are missing
    sorted_lats = [col for col, score in sorted(lat_scores.items(), key=lambda item: item[1], reverse=True) if score >= COORDINATE_THRESHOLD]
    sorted_lons = [col for col, score in sorted(lon_scores.items(), key=lambda item: item[1], reverse=True) if score >= COORDINATE_THRESHOLD]

    for lat_col in sorted_lats[:3]:
        for lon_col in sorted_lons[:3]:
            if lat_col == lon_col:
                continue
            candidate = CoordinateCandidate(
                lat_column=lat_col,
                lon_column=lon_col,
                lat_valid_ratio=lat_scores.get(lat_col, 0.0),
                lon_valid_ratio=lon_scores.get(lon_col, 0.0),
            )
            if candidate not in candidates:
                candidates.append(candidate)

    return candidates


def validate_coordinates(df: pd.DataFrame, lat_col: str, lon_col: str) -> dict:
    lat_series = pd.to_numeric(df[lat_col], errors="coerce")
    lon_series = pd.to_numeric(df[lon_col], errors="coerce")

    valid_mask = lat_series.between(-90, 90) & lon_series.between(-180, 180)
    valid_points = int(valid_mask.sum())
    total_points = int(len(df))
    invalid_points = total_points - valid_points

    return {
        "valid_points": valid_points,
        "invalid_points": invalid_points,
        "valid_ratio": (valid_points / total_points) if total_points else 0.0,
    }


def create_geospatial_figure(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    color_col: Optional[str] = None,
) -> go.Figure:
    lat_series = pd.to_numeric(df[lat_col], errors="coerce")
    lon_series = pd.to_numeric(df[lon_col], errors="coerce")
    clean_df = df.copy()
    clean_df[lat_col] = lat_series
    clean_df[lon_col] = lon_series

    clean_df = clean_df.dropna(subset=[lat_col, lon_col])
    if clean_df.empty:
        raise ValueError("No valid coordinate pairs found after cleaning")

    fig = px.scatter_geo(
        clean_df,
        lat=lat_col,
        lon=lon_col,
        color=color_col if color_col and color_col in clean_df.columns else None,
        hover_data={col: True for col in clean_df.columns if col not in {lat_col, lon_col}},
        projection="natural earth",
    )
    fig.update_layout(
        title=f"Geospatial Distribution ({lat_col} vs {lon_col})",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def _valid_ratio(series: pd.Series, low: float, high: float) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.empty:
        return 0.0
    valid = numeric.between(low, high)
    total = int(valid.count())
    return float(valid.sum() / total) if total else 0.0
