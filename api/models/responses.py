"""Pydantic response schemas."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from api.models.requests import FilterCondition


class DataLoadResponse(BaseModel):
    dataset_name: str
    rows: int
    columns: int
    column_names: List[str]


class DatabaseConnectionResponse(BaseModel):
    connected: bool
    tables: List[str]
    message: str


class DatasetOverviewResponse(BaseModel):
    name: str
    row_count: int
    column_count: int
    memory_usage_mb: float
    completeness_score: float
    column_types: Dict[str, int]
    quality_issues: List[str]
    insights: List[str]
    preview: List[Dict[str, Any]]
    describe: List[Dict[str, Any]]
    summary: str


class ColumnProfileResponse(BaseModel):
    name: str
    dtype: str
    detected_type: str
    count: int
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    numeric_stats: Optional[Dict[str, float]] = None
    categorical_stats: Optional[Dict[str, Any]] = None
    datetime_stats: Optional[Dict[str, Any]] = None
    outlier_count: Optional[int] = None
    quality_issues: Optional[List[str]] = None
    insights: List[str]


class CorrelationMatrixResponse(BaseModel):
    columns: List[str]
    matrix: List[List[float]]


class TrendResponse(BaseModel):
    column: str
    trend: str
    slope: Optional[float] = None
    r_squared: Optional[float] = None
    p_value: Optional[float] = None
    significant: Optional[bool] = None


class InsightsResponse(BaseModel):
    narrative: str
    correlations: List[Dict[str, Any]]


class CorrelationResponse(BaseModel):
    variable1: str
    variable2: str
    correlation: float
    p_value: float
    method: str
    significant: bool
    strength: str


class RegressionResponse(BaseModel):
    dependent_var: str
    independent_vars: List[str]
    coefficients: Dict[str, float]
    intercept: float
    r_squared: float
    adj_r_squared: float
    mse: float
    rmse: float
    p_values: Dict[str, float]
    insights: List[str]


class TTestResponse(BaseModel):
    group1_name: str
    group2_name: str
    group1_mean: float
    group2_mean: float
    t_statistic: float
    p_value: float
    significant: bool
    effect_size: float
    insights: List[str]


class AnovaResponse(BaseModel):
    groups: List[str]
    f_statistic: float
    p_value: float
    significant: bool
    group_means: Dict[str, float]
    insights: List[str]


class PlotlyFigureResponse(BaseModel):
    figure: Dict[str, Any]


class CurrentDatasetResponse(BaseModel):
    loaded: bool
    dataset_name: Optional[str] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    column_names: Optional[List[str]] = None
    numeric_columns: Optional[List[str]] = None
    categorical_columns: Optional[List[str]] = None
    filtered_rows: Optional[int] = None
    filters_active: bool = False
    active_filters: Optional[List[FilterCondition]] = None


class FilterOptionResponse(BaseModel):
    name: str
    detected_type: str
    searchable: bool
    categorical_values: Optional[List[str]] = None
    numeric_range: Optional[Dict[str, float]] = None
    datetime_range: Optional[Dict[str, str]] = None


class FilterOptionsResponse(BaseModel):
    columns: List[FilterOptionResponse]


class FilterApplicationResponse(BaseModel):
    row_count: int
    column_count: int
    preview: List[Dict[str, Any]]
    active_filters: List[FilterCondition]


class GeospatialResponse(BaseModel):
    figure: Dict[str, Any]
    lat_column: str
    lon_column: str
    color_column: Optional[str]
    validation: Dict[str, Any]
    warnings: List[str]
    candidates: List[Dict[str, Any]]
