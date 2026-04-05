"""
Preprocessing pipeline for the analytics engine.

Handles all data cleaning and transformation decisions that the agent
proposes to the user before any analysis begins. Every function here
returns a structured result so the orchestrator can explain what was
done and why.

Key responsibilities:
  - Missing value imputation (mean, median, mode, KNN, forward-fill, drop)
  - Categorical encoding (one-hot, label, target, ordinal)
  - Feature scaling (standard, min-max, robust, log)
  - Log-ratio transforms for compositional data (CLR, ILR, ALR)
  - Outlier handling (cap, remove, flag)
  - Duplicate removal

Read before implementing:
  - agents/mathematician.md → "Foundational Methodology Standards"
  - Aitchison (1986) for compositional transforms
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple
from unittest import result

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
# from sqlalchemy import column, null
from sklearn.preprocessing import LabelEncoder, TargetEncoder, OrdinalEncoder

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ImputationResult:
    """Result of a missing value imputation operation."""

    columns_imputed: List[str]
    strategy: str
    values_filled: Dict[str, int]        # column → number of values filled
    fill_values: Dict[str, Any]          # column → value used (for mean/median/mode)
    summary: str                         # plain-English explanation for the agent


@dataclass
class EncodingResult:
    """Result of a categorical encoding operation."""

    columns_encoded: List[str]
    strategy: str
    new_columns: List[str]               # columns added (e.g. one-hot produces many)
    dropped_columns: List[str]           # original columns removed after encoding
    mapping: Dict[str, Any]             # encoding mapping for interpretability
    summary: str


@dataclass
class ScalingResult:
    """Result of a feature scaling operation."""

    columns_scaled: List[str]
    strategy: str
    parameters: Dict[str, Dict[str, float]]  # column → {mean, std} or {min, max}
    summary: str


@dataclass
class OutlierHandlingResult:
    """Result of outlier treatment."""

    column: str
    strategy: str                        # cap | remove | flag
    outliers_found: int
    outliers_treated: int
    bounds: Tuple[float, float]          # (lower, upper)
    summary: str


@dataclass
class CompositionTransformResult:
    """Result of a log-ratio transform on compositional data."""

    transform_type: str                  # clr | ilr | alr
    input_columns: List[str]
    output_columns: List[str]
    reference_column: Optional[str]      # for ALR only
    summary: str


# ---------------------------------------------------------------------------
# Functions (to be implemented)
# ---------------------------------------------------------------------------

def impute_missing_values(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    strategy: Literal["mean", "median", "mode", "knn", "forward_fill", "drop"] = "median",
    knn_neighbors: int = 5,
) -> Tuple[pd.DataFrame, ImputationResult]:
    """
    Impute missing values in the specified columns using the given strategy.

    Args:
        df:             Input DataFrame.
        columns:        Columns to impute. None = all columns with nulls.
        strategy:       Imputation method.
        knn_neighbors:  k for KNN imputation (only used if strategy='knn').

    Returns:
        Tuple of (cleaned DataFrame, ImputationResult).

    Notes:
        KNN imputation is more accurate but much slower on large datasets.
        For time series data, forward_fill (and optionally backward_fill)
        is usually more appropriate than statistical imputation.
    """
    
    result_df = df.copy()

    if columns is None:
        columns = [col for col in df.columns if df[col].isna().any()]
    
    if not columns:
        return result_df ,ImputationResult(
            columns_imputed=[],
            strategy=strategy,
            values_filled={},
            fill_values={},
            summary="Dataset is clean, no missing values detected, nothing to impute"            
        )

    # Track null before/after
    nulls_before = {col : int(result_df[col].isna().sum()) for col in columns}
    values_filled : Dict[str , int] = {}
    fill_values : Dict[str, Any] = {}

    if strategy == "drop":
        # Drop all rows where target columns are null 
        original_length = len(result_df)
        result_df = result_df.dropna(subset=columns)
        rows_dropped = original_length - len(result_df)

        values_filled = {
                            col: nulls_before[col] - int(result_df[col].isna().sum())
                            for col in columns
                        }

        summary = (
            f"Dropped {rows_dropped} row(s) with missing values across "
            f"{len(columns)} column(s). {len(result_df)} rows remain."
        )
        
    elif strategy == "forward_fill":

        result_df[columns] = result_df[columns].ffill()
        values_filled = {
            col : nulls_before[col] - int(result_df[col].isna().sum())
            for col in columns
        }
        unfilled = sum(int(result_df[col].isna().sum()) for col in columns)
        summary = (
            f"Forward-filled {sum(values_filled.values())} missing value(s) "
            f"across {len(columns)} column(s)."
            + (f" {unfilled} leading null(s) could not be filled." if unfilled else "")
        )

    elif strategy == "knn":
        #works only on numerical columns
        numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(result_df[c])]
        skipped = set(columns) - set(numeric_cols)

        if numeric_cols:
            imputer = KNNImputer(n_neighbors=knn_neighbors)
            result_df[numeric_cols] = imputer.fit_transform(result_df[numeric_cols])

        values_filled = {
            col : nulls_before[col] - int(result_df[col].isna().sum())
            for col in columns
        }
        summary = (
            f"KNN-imputed (k={knn_neighbors}) {sum(values_filled.values())} "
            f"missing value(s) across {len(numeric_cols)} numeric column(s)."
            + (f" Skipped non-numeric: {list(skipped)}." if skipped else "")
        )


    else :
        # mean | median | mode
        # These three work the same way structurally: compute a single fill value
        # per column, then call fillna(). The difference is just how fill_val is computed.
        for col in columns:
            series = result_df[col]

            if strategy == "mean":
                if not pd.api.types.is_numeric_dtype(series):
                    continue  # mean is undefined for non-numeric — skip silently
                fill_val = series.mean()

            elif strategy == "median":
                if not pd.api.types.is_numeric_dtype(series):
                    continue
                fill_val = series.median()

            elif strategy == "mode":
                # mode() returns a Series (there can be multiple modes).
                # We take the first — the most frequent value.
                # Mode works for both numeric and categorical columns.
                mode_series = series.mode(dropna=True)
                if mode_series.empty:
                    continue  # all values are null — nothing to compute
                fill_val = mode_series.iloc[0]

            fill_values[col] = fill_val
            result_df[col] = series.fillna(fill_val)

        # Compute how many values were actually filled (before - after)
        values_filled = {
            col: nulls_before[col] - int(result_df[col].isna().sum())
            for col in fill_values
        }
        summary = (
            f"Imputed {sum(values_filled.values())} missing value(s) across "
            f"{len(values_filled)} column(s) using {strategy}."
        )

    return  result_df, ImputationResult(
        columns_imputed= list(values_filled.keys()),
        strategy=strategy,
        values_filled=values_filled,
        fill_values=fill_values,
        summary=summary
    )



def encode_categorical(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    strategy: Literal["one_hot", "label", "target", "ordinal"] = "one_hot",
    target_col: Optional[str] = None,
    ordinal_order: Optional[Dict[str, List[Any]]] = None,
    drop_first: bool = True,
) -> Tuple[pd.DataFrame, EncodingResult]:
    """
    Encode categorical columns for use in statistical models.

    Args:
        df:            Input DataFrame.
        columns:       Categorical columns to encode. None = auto-detect.
        strategy:      Encoding method.
        target_col:    Required for target encoding — the outcome variable.
        ordinal_order: Required for ordinal encoding — dict of column → ordered list.
        drop_first:    For one-hot, drop one level to avoid multicollinearity.

    Returns:
        Tuple of (encoded DataFrame, EncodingResult).

    Notes:
        One-hot is the safe default for nominal categories.
        Target encoding risks data leakage — always fit on training data only.
        Ordinal encoding only makes sense when the ordering is meaningful.
    """
    result_df = df.copy()

    if columns is None:
        columns = result_df.select_dtypes(include=["category","object"]).columns.to_list()
    
    if not columns:
        return result_df, EncodingResult(
                columns_encoded = [],
                strategy =  strategy, 
                new_columns = [],       
                dropped_columns = [],      
                mapping  = {},       
                summary = f"No categorical columns present to be encoded"
        )

    if strategy == "one_hot":
        columns_before = set(result_df.columns)

        result_df = pd.get_dummies(result_df , columns=columns, drop_first=drop_first)
        columns_after  = set(result_df.columns)
        
        new_encoded_cols = list(columns_after - columns_before)
        dropped_cols = list(columns_before - columns_after)

        return result_df, EncodingResult(
            columns_encoded=columns,
            strategy=strategy, 
            new_columns=sorted(new_encoded_cols),
            mapping={},
            summary = (f"One hot encoding {len(columns)} column(s) into"
                        f"{len(new_encoded_cols)} binary column(s)."
                        + (" Dropped first level to avoid multicollinearity." if drop_first else "")
            )
        )

    elif strategy == "label":
        label_encoder = LabelEncoder()

        mapping={}

        for col in columns:
            result_df[col] = label_encoder.fit_transform(result_df[col])
            mapping[col] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

        return result_df, EncodingResult(
            columns_encoded=columns,
            strategy=strategy,
            new_columns=[],           # no new columns — same columns, different values
            dropped_columns=[],       # nothing dropped
            mapping=mapping,
            summary=(
                f"Label encoded {len(columns)} column(s). "
                f"Each category mapped to an integer. "
                f"See mapping for the correspondence."
            )
        )

    elif strategy == "target":
        target_encoder = TargetEncoder()
        mapping={}

        for col in columns:
            mean_map = result_df.groupby(col)[target_col].mean()
            result_df[col] = result_df[col].map(mean_map)



def scale_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    strategy: Literal["standard", "minmax", "robust", "log"] = "standard",
) -> Tuple[pd.DataFrame, ScalingResult]:
    """
    Scale numeric features.

    Args:
        df:       Input DataFrame.
        columns:  Numeric columns to scale. None = all numeric columns.
        strategy: Scaling method.

    Returns:
        Tuple of (scaled DataFrame, ScalingResult).

    Notes:
        standard  → Z-score (mean=0, std=1). Assumes approximate normality.
        minmax    → [0, 1] range. Sensitive to outliers.
        robust    → uses median and IQR instead of mean/std. Best when outliers present.
        log       → log(x+1) transform. Useful for right-skewed distributions.
                    Check that all values are non-negative first.
    """
    pass


def handle_outliers(
    df: pd.DataFrame,
    column: str,
    strategy: Literal["cap", "remove", "flag"] = "cap",
    method: Literal["iqr", "zscore"] = "iqr",
    threshold: float = 1.5,
) -> Tuple[pd.DataFrame, OutlierHandlingResult]:
    """
    Treat outliers in a single column.

    Args:
        df:        Input DataFrame.
        column:    Column to process.
        strategy:  How to handle outliers — cap (Winsorise), remove rows, or flag.
        method:    Detection method: IQR (multiplier) or Z-score (std deviations).
        threshold: IQR multiplier (1.5 standard, 3.0 conservative) or Z-score cutoff.

    Returns:
        Tuple of (treated DataFrame, OutlierHandlingResult).
    """
    pass


def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: Literal["first", "last"] = "first",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Remove duplicate rows and return a summary of what was removed.
    """
    pass


def apply_clr_transform(
    df: pd.DataFrame,
    columns: List[str],
) -> Tuple[pd.DataFrame, CompositionTransformResult]:
    """
    Apply the Centered Log-Ratio (CLR) transform to compositional columns.

    CLR(x)_i = log(x_i / geometric_mean(x))

    The CLR maps from the simplex to real space, making Euclidean geometry
    valid. Note: the CLR covariance matrix is singular (rank D-1) — use ILR
    for PCA and regression.

    Args:
        df:      DataFrame containing the composition.
        columns: The parts of the composition (must sum to a constant).

    Returns:
        Tuple of (transformed DataFrame, CompositionTransformResult).
    """
    pass


def apply_ilr_transform(
    df: pd.DataFrame,
    columns: List[str],
) -> Tuple[pd.DataFrame, CompositionTransformResult]:
    """
    Apply the Isometric Log-Ratio (ILR) transform to compositional columns.

    ILR maps D parts to D-1 real coordinates using an orthonormal basis
    on the simplex (sequential binary partition). The resulting coordinates
    have a full-rank covariance matrix — suitable for PCA, regression,
    clustering, and all multivariate methods.

    Args:
        df:      DataFrame containing the composition.
        columns: The parts of the composition.

    Returns:
        Tuple of (transformed DataFrame, CompositionTransformResult).
    """
    pass
