"""
Data profiling engine for automated exploratory data analysis.

Analyzes datasets to extract statistical summaries, detect data types,
identify quality issues, and generate comprehensive profiles.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Literal
import pandas as pd
import numpy as np
from scipy import stats


ColumnType = Literal["numeric", "categorical", "datetime", "boolean", "text", "unknown"]


@dataclass
class ColumnProfile:
    """Statistical profile for a single column."""
    
    name: str
    dtype: str
    detected_type: ColumnType
    
    # Basic stats
    count: int
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    
    # Type-specific stats
    numeric_stats: Optional[Dict[str, float]] = None
    categorical_stats: Optional[Dict[str, Any]] = None
    datetime_stats: Optional[Dict[str, Any]] = None
    
    # Data quality
    outliers: Optional[List[Any]] = None
    quality_issues: Optional[List[str]] = None


@dataclass
class DatasetProfile:
    """Comprehensive profile for an entire dataset."""
    
    name: str
    row_count: int
    column_count: int
    memory_usage: int  # bytes
    
    columns: List[ColumnProfile]
    column_types: Dict[ColumnType, int]
    
    # Dataset-level insights
    completeness_score: float  # 0-1, percentage of non-null values
    quality_issues: List[str]
    correlations: Optional[pd.DataFrame] = None


def detect_column_type(series: pd.Series) -> ColumnType:
    """
    Detect semantic column type beyond pandas dtype.
    
    Args:
        series: Pandas series to analyze
        
    Returns:
        Detected column type
    """
    # Already datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    
    # Boolean
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    
    # Numeric
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    
    # Check if it's categorical (low cardinality strings)
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
        non_null = series.dropna()
        if len(non_null) == 0:
            return "unknown"
        
        unique_ratio = non_null.nunique() / len(non_null)
        
        # Low cardinality = categorical
        if unique_ratio < 0.05 or non_null.nunique() < 50:
            return "categorical"
        
        # High cardinality = text
        return "text"
    
    return "unknown"


def profile_column(series: pd.Series) -> ColumnProfile:
    """
    Generate comprehensive profile for a single column.
    
    Args:
        series: Pandas series to profile
        
    Returns:
        ColumnProfile with statistical summary
    """
    name = series.name
    count = len(series)
    null_count = series.isna().sum()
    null_pct = (null_count / count * 100) if count > 0 else 0
    
    non_null = series.dropna()
    unique_count = non_null.nunique()
    unique_pct = (unique_count / count * 100) if count > 0 else 0
    
    detected_type = detect_column_type(series)
    
    # Type-specific profiling
    numeric_stats = None
    categorical_stats = None
    datetime_stats = None
    outliers = None
    quality_issues = []
    
    if detected_type == "numeric":
        numeric_stats = {
            "mean": float(non_null.mean()),
            "median": float(non_null.median()),
            "std": float(non_null.std()),
            "min": float(non_null.min()),
            "max": float(non_null.max()),
            "q25": float(non_null.quantile(0.25)),
            "q75": float(non_null.quantile(0.75)),
            "skewness": float(stats.skew(non_null)),
            "kurtosis": float(stats.kurtosis(non_null))
        }
        
        # Detect outliers using IQR method
        outliers = detect_outliers_iqr(non_null)
        if len(outliers) > 0:
            quality_issues.append(f"{len(outliers)} outliers detected")
    
    elif detected_type == "categorical":
        value_counts = non_null.value_counts()
        categorical_stats = {
            "top_values": value_counts.head(10).to_dict(),
            "mode": value_counts.index[0] if len(value_counts) > 0 else None,
            "mode_frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            "entropy": float(stats.entropy(value_counts))
        }
    
    elif detected_type == "datetime":
        datetime_stats = {
            "min": non_null.min(),
            "max": non_null.max(),
            "range_days": (non_null.max() - non_null.min()).days if len(non_null) > 0 else 0
        }
    
    # Data quality checks
    if null_pct > 50:
        quality_issues.append(f"High null percentage: {null_pct:.1f}%")
    
    if unique_count == count and count > 1:
        quality_issues.append("All values are unique (possible ID column)")
    
    if unique_count == 1:
        quality_issues.append("Only one unique value (constant column)")
    
    return ColumnProfile(
        name=name,
        dtype=str(series.dtype),
        detected_type=detected_type,
        count=count,
        null_count=null_count,
        null_percentage=null_pct,
        unique_count=unique_count,
        unique_percentage=unique_pct,
        numeric_stats=numeric_stats,
        categorical_stats=categorical_stats,
        datetime_stats=datetime_stats,
        outliers=outliers,
        quality_issues=quality_issues if quality_issues else None
    )


def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> List[float]:
    """
    Detect outliers using the IQR (Interquartile Range) method.
    
    Args:
        series: Numeric series to analyze
        multiplier: IQR multiplier (default 1.5)
        
    Returns:
        List of outlier values
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return outliers.tolist()[:100]  # Limit to first 100 outliers


def profile_dataset(df: pd.DataFrame, name: str = "Dataset") -> DatasetProfile:
    """
    Generate comprehensive profile for an entire dataset.
    
    Args:
        df: DataFrame to profile
        name: Dataset name for reporting
        
    Returns:
        DatasetProfile with full analysis
    """
    # Profile each column
    column_profiles = [profile_column(df[col]) for col in df.columns]
    
    # Count column types
    type_counts: Dict[ColumnType, int] = {}
    for profile in column_profiles:
        type_counts[profile.detected_type] = type_counts.get(profile.detected_type, 0) + 1
    
    # Calculate completeness
    total_cells = df.shape[0] * df.shape[1]
    null_cells = df.isna().sum().sum()
    completeness = 1 - (null_cells / total_cells) if total_cells > 0 else 0
    
    # Dataset-level quality issues
    quality_issues = []
    
    if completeness < 0.8:
        quality_issues.append(f"Low data completeness: {completeness*100:.1f}%")
    
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        quality_issues.append(f"{duplicate_rows} duplicate rows detected")
    
    # Calculate correlations for numeric columns
    numeric_cols = [p.name for p in column_profiles if p.detected_type == "numeric"]
    correlations = None
    if len(numeric_cols) > 1:
        correlations = df[numeric_cols].corr()
    
    return DatasetProfile(
        name=name,
        row_count=len(df),
        column_count=len(df.columns),
        memory_usage=df.memory_usage(deep=True).sum(),
        columns=column_profiles,
        column_types=type_counts,
        completeness_score=completeness,
        quality_issues=quality_issues,
        correlations=correlations
    )


def identify_data_quality_issues(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Scan dataset for common data quality problems.
    """
    issues = []

    # Check for completely empty columns
    for col in df.columns:
        if df[col].isna().all():
            issues.append({
                "column": col,
                "issue": "empty_column",
                "severity": "high",
                "description": "Column contains only null values"
            })

    # Check for high cardinality columns (categorical OR text)
    for col in df.columns:
        col_type = detect_column_type(df[col])
        unique_ratio = df[col].nunique() / len(df)

        if col_type in ("categorical", "text") and unique_ratio > 0.9:
            issues.append({
                "column": col,
                "issue": "high_cardinality",
                "severity": "medium",
                "description": f"Column has {unique_ratio*100:.1f}% unique values"
            })

    # Check for mixed types in object columns
    for col in df.select_dtypes(include=["object"]).columns:
        type_set = set(type(x).__name__ for x in df[col].dropna())
        if len(type_set) > 1:
            issues.append({
                "column": col,
                "issue": "mixed_types",
                "severity": "high",
                "description": f"Column contains mixed types: {', '.join(type_set)}"
            })

    return issues

