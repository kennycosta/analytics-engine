"""
Data source endpoints: file upload, DB connect, table load, query, sample data.
"""

import io
from typing import Any, List, Dict

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
import pandas as pd
import numpy as np

from api.dependencies import get_session_data, require_data
from api.session_store import SessionData
from api.models.requests import (
    DatabaseConnectRequest,
    LoadTableRequest,
    RunQueryRequest,
    LoadSampleRequest,
    ApplyFiltersRequest,
)
from api.models.responses import (
    DataLoadResponse,
    DatabaseConnectionResponse,
    CurrentDatasetResponse,
    FilterOptionsResponse,
    FilterOptionResponse,
    FilterApplicationResponse,
)
from core.profiling import profile_dataset, detect_column_type
from core.filtering import apply_filters
from db.connection import DatabaseClient
from db.introspection import get_tables
from db.loader import load_table
from db.query import run_safe_query

router = APIRouter(prefix="/api/data", tags=["data"])


def _persist(session: SessionData, df: pd.DataFrame, name: str) -> DataLoadResponse:
    """Store the dataframe in the session and profile it."""
    session.raw_data = df
    session.current_data = df
    session.dataset_name = name
    session.dataset_profile = profile_dataset(df, name)
    session.active_filters = []
    return DataLoadResponse(
        dataset_name=name,
        rows=len(df),
        columns=len(df.columns),
        column_names=df.columns.tolist(),
    )


@router.post("/upload", response_model=DataLoadResponse)
async def upload_file(
    file: UploadFile = File(...),
    session: SessionData = Depends(get_session_data),
):
    """Upload a CSV or Excel file."""
    content = await file.read()
    filename = file.filename or "upload"

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use CSV or Excel.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {exc}")

    return _persist(session, df, filename)


@router.post("/connect-database", response_model=DatabaseConnectionResponse)
def connect_database(
    body: DatabaseConnectRequest,
    session: SessionData = Depends(get_session_data),
):
    """Connect to a SQL Server database."""
    try:
        cfg = session.config.db
        cfg.host = body.server
        cfg.database = body.database
        db_client = DatabaseClient(cfg)
        if not db_client.test_connection():
            raise RuntimeError("Connection test failed")
        session.db_client = db_client
        tables = get_tables(db_client)
        session.available_tables = tables
        return DatabaseConnectionResponse(connected=True, tables=tables, message="Connected successfully")
    except Exception as exc:
        session.db_client = None
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/load-table", response_model=DataLoadResponse)
def load_table_endpoint(
    body: LoadTableRequest,
    session: SessionData = Depends(get_session_data),
):
    """Load a table from the connected database."""
    if session.db_client is None:
        raise HTTPException(status_code=400, detail="Not connected to a database")
    try:
        df = load_table(session.db_client, table_name=body.table_name, limit=body.limit)
        return _persist(session, df, body.table_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/run-query", response_model=DataLoadResponse)
def run_query_endpoint(
    body: RunQueryRequest,
    session: SessionData = Depends(get_session_data),
):
    """Execute a read-only SQL query."""
    if session.db_client is None:
        raise HTTPException(status_code=400, detail="Not connected to a database")
    try:
        df = run_safe_query(session.db_client, body.query)
        return _persist(session, df, "SQL Query")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/load-sample", response_model=DataLoadResponse)
def load_sample(
    body: LoadSampleRequest,
    session: SessionData = Depends(get_session_data),
):
    """Load a built-in sample dataset."""
    np.random.seed(42)
    sample_type = body.sample_type.lower()

    if sample_type == "sales":
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=100),
            "revenue": np.random.normal(10_000, 2_000, 100),
            "units": np.random.poisson(50, 100),
            "region": np.random.choice(["N", "S", "E", "W"], 100),
        })
        name = "Sample Sales"
    elif sample_type == "customer":
        df = pd.DataFrame({
            "age": np.random.randint(18, 70, 200),
            "income": np.random.lognormal(10.5, 0.5, 200),
            "segment": np.random.choice(["A", "B", "C"], 200),
        })
        name = "Sample Customer"
    elif sample_type == "random":
        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 150),
            "y": np.random.normal(5, 2, 150),
            "group": np.random.choice(["A", "B", "C"], 150),
        })
        name = "Random Data"
    else:
        raise HTTPException(status_code=400, detail=f"Unknown sample type: {body.sample_type}")

    return _persist(session, df, name)


@router.get("/current-dataset", response_model=CurrentDatasetResponse)
def current_dataset(session: SessionData = Depends(get_session_data)):
    """Return metadata about the currently loaded dataset."""
    if session.current_data is None:
        return CurrentDatasetResponse(loaded=False)

    df = session.current_data
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [
        c for c in df.columns
        if detect_column_type(df[c]) == "categorical"
    ]

    return CurrentDatasetResponse(
        loaded=True,
        dataset_name=session.dataset_name,
        rows=len(session.raw_data) if session.raw_data is not None else len(df),
        filtered_rows=len(df),
        columns=len(df.columns),
        column_names=df.columns.tolist(),
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        filters_active=bool(session.active_filters),
        active_filters=session.active_filters or None,
    )


@router.get("/filter-options", response_model=FilterOptionsResponse)
def filter_options(session: SessionData = Depends(require_data)):
    """Return filter metadata for each column based on the active dataset."""
    df = session.current_data
    columns = []
    for col in df.columns:
        detected = detect_column_type(df[col])
        option = FilterOptionResponse(
            name=col,
            detected_type=detected,
            searchable=detected in ("categorical", "text"),
        )

        if detected == "numeric":
            numeric = pd.to_numeric(df[col], errors="coerce")
            option.numeric_range = {
                "min": float(numeric.min()) if numeric.notna().any() else 0.0,
                "max": float(numeric.max()) if numeric.notna().any() else 0.0,
            }
        elif detected == "datetime":
            dt_series = pd.to_datetime(df[col], errors="coerce")
            option.datetime_range = {
                "start": dt_series.min().isoformat() if dt_series.notna().any() else None,
                "end": dt_series.max().isoformat() if dt_series.notna().any() else None,
            }
        else:
            values = (
                df[col]
                .dropna()
                .astype(str)
                .value_counts()
                .head(50)
                .index
                .tolist()
            )
            option.categorical_values = values

        columns.append(option)

    return FilterOptionsResponse(columns=columns)


@router.post("/filters/apply", response_model=FilterApplicationResponse)
def apply_filter_set(
    body: ApplyFiltersRequest,
    session: SessionData = Depends(require_data),
):
    """Apply cascading filters to the dataset and refresh the active profile."""
    if session.raw_data is None:
        raise HTTPException(status_code=400, detail="No dataset is available for filtering")

    filter_payload = [f.dict() for f in body.filters]
    try:
        filtered_df = apply_filters(session.raw_data, filter_payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    session.current_data = filtered_df
    profile_name = session.dataset_name or "Dataset"
    if filter_payload:
        profile_name = f"{profile_name} (filtered)"
    session.dataset_profile = profile_dataset(filtered_df, profile_name)
    session.active_filters = filter_payload

    preview = _preview_rows(filtered_df)
    return FilterApplicationResponse(
        row_count=len(filtered_df),
        column_count=filtered_df.shape[1],
        preview=preview,
        active_filters=body.filters,
    )


@router.post("/filters/reset", response_model=FilterApplicationResponse)
def reset_filters(session: SessionData = Depends(require_data)):
    """Clear all applied filters and restore the raw dataset view."""
    if session.raw_data is None:
        raise HTTPException(status_code=400, detail="No dataset is available for filtering")

    session.current_data = session.raw_data
    session.dataset_profile = profile_dataset(session.raw_data, session.dataset_name or "Dataset")
    session.active_filters = []
    preview = _preview_rows(session.current_data)
    return FilterApplicationResponse(
        row_count=len(session.current_data),
        column_count=session.current_data.shape[1],
        preview=preview,
        active_filters=[],
    )


def _preview_rows(df: pd.DataFrame, limit: int = 20) -> List[Dict[str, Any]]:
    preview = df.head(limit).copy()
    datetime_cols = preview.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
    for col in datetime_cols:
        preview[col] = preview[col].astype(str)
    return preview.where(preview.notna(), None).to_dict(orient="records")
