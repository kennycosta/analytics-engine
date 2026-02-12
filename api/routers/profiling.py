"""
Profiling endpoints: dataset overview and per-column profiles.
"""

from typing import Any, Dict, List

from fastapi import APIRouter, Depends
import pandas as pd

from api.dependencies import require_data
from api.session_store import SessionData
from api.models.responses import DatasetOverviewResponse, ColumnProfileResponse
from core.insights import InsightGenerator
from core.profiling import profile_column

router = APIRouter(prefix="/api/profile", tags=["profiling"])


@router.get("/overview", response_model=DatasetOverviewResponse)
def overview(session: SessionData = Depends(require_data)):
    """Return full dataset overview with insights."""
    profile = session.dataset_profile
    df = session.current_data
    insights = InsightGenerator.generate_dataset_overview(profile)
    summary = InsightGenerator.generate_dataset_summary(profile)
    describe_rows = _describe_dataframe(df)

    preview = df.head(20).copy()
    # Convert timestamps to strings for JSON
    for col in preview.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns:
        preview[col] = preview[col].astype(str)

    return DatasetOverviewResponse(
        name=profile.name,
        row_count=profile.row_count,
        column_count=profile.column_count,
        memory_usage_mb=round(profile.memory_usage / 1024 / 1024, 2),
        completeness_score=profile.completeness_score,
        column_types={str(k): v for k, v in profile.column_types.items()},
        quality_issues=profile.quality_issues,
        insights=insights,
        preview=preview.where(preview.notna(), None).to_dict(orient="records"),
        describe=describe_rows,
        summary=summary,
    )


@router.get("/columns", response_model=List[ColumnProfileResponse])
def all_columns(session: SessionData = Depends(require_data)):
    """Return profiles for all columns."""
    df = session.current_data
    results = []
    for col in df.columns:
        cp = profile_column(df[col])
        insights = InsightGenerator.generate_column_insights(cp)
        results.append(_to_response(cp, insights))
    return results


@router.get("/column/{name}", response_model=ColumnProfileResponse)
def single_column(name: str, session: SessionData = Depends(require_data)):
    """Return profile for a single column."""
    df = session.current_data
    if name not in df.columns:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Column '{name}' not found")
    cp = profile_column(df[name])
    insights = InsightGenerator.generate_column_insights(cp)
    return _to_response(cp, insights)


def _to_response(cp, insights: List[str]) -> ColumnProfileResponse:
    datetime_stats = None
    if cp.datetime_stats:
        datetime_stats = {k: str(v) for k, v in cp.datetime_stats.items()}
    return ColumnProfileResponse(
        name=cp.name,
        dtype=cp.dtype,
        detected_type=cp.detected_type,
        count=cp.count,
        null_count=cp.null_count,
        null_percentage=cp.null_percentage,
        unique_count=cp.unique_count,
        unique_percentage=cp.unique_percentage,
        numeric_stats=cp.numeric_stats,
        categorical_stats=cp.categorical_stats,
        datetime_stats=datetime_stats,
        outlier_count=len(cp.outliers) if cp.outliers else None,
        quality_issues=cp.quality_issues,
        insights=insights,
    )


def _describe_dataframe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.empty:
        return []

    desc = df.describe(include="all", datetime_is_numeric=False).transpose()
    desc = desc.where(desc.notna(), None)

    records: List[Dict[str, Any]] = []
    for column, stats in desc.iterrows():
        serialized = {key: _serialize_value(value) for key, value in stats.items()}
        records.append({"column": column, "stats": serialized})
    return records


def _serialize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)
