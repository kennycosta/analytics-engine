"""Pydantic request schemas."""

from typing import Any, List, Literal, Optional
from pydantic import BaseModel


class DatabaseConnectRequest(BaseModel):
    server: str
    database: str


class LoadTableRequest(BaseModel):
    table_name: str
    limit: Optional[int] = 5000


class RunQueryRequest(BaseModel):
    query: str


class LoadSampleRequest(BaseModel):
    sample_type: str  # "sales", "customer", "random"


class TrendRequest(BaseModel):
    column: str


class CorrelationRequest(BaseModel):
    col1: str
    col2: str
    method: str = "pearson"


class FilterCondition(BaseModel):
    column: str
    type: Literal["numeric", "categorical", "datetime", "boolean", "text"]
    operator: str
    value: Optional[Any] = None
    values: Optional[List[Any]] = None
    range: Optional[List[Any]] = None


class ApplyFiltersRequest(BaseModel):
    filters: List[FilterCondition]


class GeospatialRequest(BaseModel):
    lat_column: Optional[str] = None
    lon_column: Optional[str] = None
    color_column: Optional[str] = None


class TimeSeriesRequest(BaseModel):
    value_columns: List[str]
    x_column: Optional[str] = None
    chart_type: Literal["line", "area", "stacked"] = "line"
    rolling_window: Optional[int] = None


class RegressionRequest(BaseModel):
    dependent_var: str
    independent_vars: List[str]


class TTestRequest(BaseModel):
    numeric_col: str
    group_col: str
    group1_value: str
    group2_value: str


class AnovaRequest(BaseModel):
    numeric_col: str
    group_col: str
