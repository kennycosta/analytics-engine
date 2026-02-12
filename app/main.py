"""
AI Analytics Engine - Streamlit Application

Interactive UI layer for the analytics platform.
All database access is routed through the db abstraction layer.
"""

from dataclasses import dataclass, field
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats as scipy_stats
import streamlit as st
import streamlit.components.v1 as components
import sys

try:
    from pyproj import CRS, Transformer
except ImportError:  # pragma: no cover - optional dependency
    CRS = None
    Transformer = None

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Config
from db.connection import DatabaseClient
from db.introspection import get_tables
from db.loader import load_table
from db.query import run_safe_query

from core.profiling import DatasetProfile, profile_dataset, detect_outliers_iqr
from core.statistics import (
    correlation_matrix,
    calculate_correlation,
    detect_trend,
    linear_regression_analysis,
    independent_t_test,
    anova_test,
    welch_t_test,
    paired_t_test,
    mann_whitney_u_test,
    wilcoxon_signed_rank_test,
    chi_square_test,
    two_way_anova,
    tukey_hsd_posthoc,
    levene_test,
    detect_outliers_zscore,
)
from core.insights import InsightGenerator
from core.visualizations import (
    create_histogram_distribution,
    create_violin_distribution,
    create_ecdf_plot,
    create_box_distribution,
    create_correlation_heatmap,
    create_scatter_plot,
    create_box_plot_by_group,
    create_count_plot,
    create_regression_plot,
    create_residual_diagnostic_plot,
)


# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Analytics Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------
# Cached helpers
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_profile_dataset(df: pd.DataFrame, name: str) -> DatasetProfile:
    return profile_dataset(df, name)


# Correlation cache managed manually to avoid hashing large DataFrames each rerun.
def get_cached_correlation(
    df: pd.DataFrame,
    columns: Tuple[str, ...],
    method: str,
) -> pd.DataFrame:
    cache: Dict[Tuple[int, int, int, Tuple[str, ...], str], pd.DataFrame] = st.session_state.setdefault(
        "correlation_cache",
        {},
    )
    signature = (
        st.session_state.get(DATASET_VERSION_KEY, 0),
        st.session_state.get(FILTER_VERSION_KEY, 0),
        len(df),
        columns,
        method,
    )
    if signature not in cache:
        cache[signature] = correlation_matrix(df, list(columns), method=method)
    return cache[signature]


def get_cached_distribution_figure(
    series: pd.Series,
    *,
    axis_label: str,
    log_enabled: bool,
    distribution_type: str,
    builder: Callable[[], go.Figure],
    outlier_points: Optional[pd.Series] = None,
    bins: Optional[int] = None,
) -> go.Figure:
    cache: Dict[Tuple[Any, ...], go.Figure] = st.session_state.setdefault("distribution_figure_cache", {})

    clean_series = series.dropna()
    series_hash = int(pd.util.hash_pandas_object(clean_series, index=False).sum()) if not clean_series.empty else 0
    outlier_hash = 0
    if outlier_points is not None and not outlier_points.empty:
        outlier_hash = int(pd.util.hash_pandas_object(outlier_points.dropna(), index=False).sum())

    signature = (
        st.session_state.get(DATASET_VERSION_KEY, 0),
        st.session_state.get(FILTER_VERSION_KEY, 0),
        st.session_state.get(FILTER_SIGNATURE_KEY),
        series.name,
        axis_label,
        log_enabled,
        distribution_type,
        len(clean_series),
        series_hash,
        outlier_hash,
        bins,
    )

    if signature not in cache:
        cache[signature] = builder()

    return cache[signature]


# ------------------------------------------------------------------
# Session state utilities
# ------------------------------------------------------------------
ANALYSIS_STATE_KEY = "analysis_store"
TAB_TITLES = ["Overview", "Relationships", "Trends", "Geospatial", "Insights"]
NAV_RADIO_KEY = "primary_navigation"
FILTERED_DATA_KEY = "filtered_data"
FILTERS_ACTIVE_KEY = "filters_active"
DATASET_VERSION_KEY = "dataset_version"
FILTER_VERSION_KEY = "filter_version"
FILTER_SIGNATURE_KEY = "filter_signature"

STATUS_COLORS = {
    "significant": "green",
    "neutral": "gray",
    "warning": "orange",
    "critical": "red",
}


@dataclass
class VisualFilterContext:
    dataframe: pd.DataFrame
    advanced_enabled: bool = False
    outlier_column: Optional[str] = None
    outlier_method: Optional[str] = None
    outlier_values: Optional[pd.Series] = None
    zscore_threshold: Optional[float] = None
    log_columns: Dict[str, bool] = field(default_factory=dict)


def status_badge(level: str, message: str) -> str:
    color = STATUS_COLORS.get(level, "gray")
    return f":{color}[{message}]"


def ensure_multiselect_state(key: str, options: List[Any]) -> List[Any]:
    previous = st.session_state.get(key, None)

    if previous is None:
        sanitized = options.copy()
    else:
        sanitized = [opt for opt in previous if opt in options]
        if not sanitized:
            if previous == []:
                sanitized = []
            else:
                sanitized = options.copy() if options else []

    st.session_state[key] = sanitized
    return sanitized


def ensure_range_state(key: str, bounds: Tuple[Any, Any]) -> Tuple[Any, Any]:
    low, high = bounds
    bounds_key = f"{key}_bounds"
    current = st.session_state.get(key)
    stored_bounds = st.session_state.get(bounds_key)

    if current is None:
        st.session_state[key] = bounds
        st.session_state[bounds_key] = bounds
        return bounds

    current_low, current_high = current
    if stored_bounds != bounds:
        clamped_low = min(max(current_low, low), high)
        clamped_high = min(max(current_high, clamped_low), high)
        new_range = (clamped_low, clamped_high)
        st.session_state[key] = new_range
        st.session_state[bounds_key] = bounds
        return new_range

    if stored_bounds is None:
        st.session_state[bounds_key] = bounds

    return current


def stable_selectbox(
    label: str,
    options: List[Any],
    *,
    key: str,
    default_index: int = 0,
    format_func=None,
    help: Optional[str] = None,
) -> Any:
    option_list = list(options)
    if option_list and key not in st.session_state:
        st.session_state[key] = option_list[min(default_index, len(option_list) - 1)]
    if option_list and st.session_state.get(key) not in option_list:
        st.session_state[key] = option_list[0]
    index = option_list.index(st.session_state[key]) if option_list else 0
    selectbox_kwargs = {
        "label": label,
        "options": option_list,
        "index": index,
        "key": key,
        "help": help,
    }
    if format_func is not None:
        selectbox_kwargs["format_func"] = format_func
    return st.selectbox(**selectbox_kwargs)


def apply_log_scale_to_series(series: pd.Series) -> Tuple[pd.Series, int]:
    mask = series > 0
    removed = int((~mask).sum())
    transformed = series[mask]
    if not transformed.empty:
        transformed = np.log1p(transformed)
    return transformed, removed


def apply_log_scale_to_dataframe(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, int]:
    mask = df[column] > 0
    removed = int((~mask).sum())
    transformed_df = df[mask].copy()
    if not transformed_df.empty:
        transformed_df[column] = np.log1p(transformed_df[column])
    return transformed_df, removed


def limit_visual_records(df: pd.DataFrame, max_rows: int = 3000) -> pd.DataFrame:
    if df is None or len(df) <= max_rows:
        return df
    return df.sample(max_rows, random_state=42)


def limit_visual_series(series: pd.Series, max_rows: int = 3000) -> pd.Series:
    if series is None or len(series) <= max_rows:
        return series
    return series.sample(max_rows, random_state=42)


def init_session_state() -> None:
    if "config" not in st.session_state:
        st.session_state.config = Config.load()

    if "db_client" not in st.session_state:
        st.session_state.db_client = None

    if "available_tables" not in st.session_state:
        st.session_state.available_tables = []

    if "current_data" not in st.session_state:
        st.session_state.current_data = None

    if "dataset_profile" not in st.session_state:
        st.session_state.dataset_profile = None

    if "dataset_name" not in st.session_state:
        st.session_state.dataset_name = None

    if ANALYSIS_STATE_KEY not in st.session_state:
        reset_analysis_state()

    if NAV_RADIO_KEY not in st.session_state:
        st.session_state[NAV_RADIO_KEY] = TAB_TITLES[0]

    if FILTERED_DATA_KEY not in st.session_state:
        st.session_state[FILTERED_DATA_KEY] = None

    if FILTERS_ACTIVE_KEY not in st.session_state:
        st.session_state[FILTERS_ACTIVE_KEY] = False

    if DATASET_VERSION_KEY not in st.session_state:
        st.session_state[DATASET_VERSION_KEY] = 0

    if FILTER_VERSION_KEY not in st.session_state:
        st.session_state[FILTER_VERSION_KEY] = 0

    if FILTER_SIGNATURE_KEY not in st.session_state:
        st.session_state[FILTER_SIGNATURE_KEY] = None


def reset_analysis_state() -> None:
    st.session_state[ANALYSIS_STATE_KEY] = {
        "correlations": [],
        "regression": None,
        "trend": None,
        "tests": [],
    }


def persist_dataset(df: pd.DataFrame, dataset_name: str) -> None:
    st.session_state.current_data = df
    st.session_state.dataset_name = dataset_name
    reset_analysis_state()
    st.session_state[DATASET_VERSION_KEY] += 1
    st.session_state[FILTER_VERSION_KEY] = 0
    st.session_state[FILTER_SIGNATURE_KEY] = None

    with st.spinner("Profiling dataset…"):
        base_profile = cached_profile_dataset(df, dataset_name)

    st.session_state.dataset_profile = base_profile
    st.session_state[FILTERED_DATA_KEY] = df
    st.session_state[FILTERS_ACTIVE_KEY] = False
    st.session_state["correlation_cache"] = {}
    st.session_state["distribution_figure_cache"] = {}
    st.session_state["geo_projection_cache"] = {}

    for key in list(st.session_state.keys()):
        if key.startswith("filter_cat_") or key.startswith("filter_num_") or key.startswith("filter_dt_"):
            del st.session_state[key]


# ------------------------------------------------------------------
# Column helpers
# ------------------------------------------------------------------
def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def get_time_columns(df: pd.DataFrame) -> List[str]:
    datetime_cols = get_datetime_columns(df)
    ordered_numeric = [col for col in get_numeric_columns(df) if not np.issubdtype(df[col].dtype, bool)]
    return datetime_cols + ordered_numeric


def get_datetime_columns(df: pd.DataFrame) -> List[str]:
    datetime_cols: List[str] = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_datetime64tz_dtype(series):
            datetime_cols.append(column)
    return datetime_cols


def get_active_dataframe() -> Optional[pd.DataFrame]:
    return st.session_state.get(FILTERED_DATA_KEY)


def get_active_profile() -> Optional[DatasetProfile]:
    return st.session_state.get("dataset_profile")


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
def render_sidebar() -> None:
    st.sidebar.title("Analytics Engine")
    st.sidebar.markdown("---")

    source_type = st.sidebar.radio(
        "Data source",
        ["Upload file", "SQL Database", "Sample data"],
        key="data_source_selector",
    )

    df = None
    dataset_name = None

    # ---------------- File upload ----------------
    if source_type == "Upload file":
        uploaded = st.sidebar.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "xlsx", "xls"],
            key="uploaded_file",
        )

        if uploaded:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)

            dataset_name = uploaded.name
            st.sidebar.success(f"Loaded {len(df)} rows")

    # ---------------- SQL Database ----------------
    elif source_type == "SQL Database":
        cfg = st.session_state.config.db

        st.sidebar.subheader("Connection")
        server = st.sidebar.text_input("Server", value=cfg.host, key="db_server_input")
        database = st.sidebar.text_input("Database", value=cfg.database, key="db_name_input")

        if st.sidebar.button("Connect", key="connect_db_btn"):
            try:
                cfg.host = server
                cfg.database = database
                db_client = DatabaseClient(cfg)

                if not db_client.test_connection():
                    raise RuntimeError("Connection test failed")

                st.session_state.db_client = db_client
                st.session_state.available_tables = get_tables(db_client)

                st.sidebar.success("Connected successfully")
            except Exception as exc:  # pragma: no cover - UI feedback
                st.sidebar.error(str(exc))
                st.session_state.db_client = None

        if st.session_state.db_client:
            st.sidebar.markdown("---")
            st.sidebar.subheader("Load data")

            mode = st.sidebar.radio(
                "Load mode",
                ["Table", "Custom SQL (read-only)"],
                key="data_load_mode",
            )

            row_limit = st.sidebar.number_input(
                "Row limit",
                min_value=100,
                max_value=100_000,
                value=5_000,
                step=500,
                key="row_limit_input",
            )

            if mode == "Table":
                table = st.sidebar.selectbox(
                    "Table",
                    st.session_state.available_tables,
                    key="table_selector",
                )

                if st.sidebar.button("Load table", key="load_table_btn"):
                    df = load_table(
                        st.session_state.db_client,
                        table_name=table,
                        limit=row_limit,
                    )
                    dataset_name = table
                    st.sidebar.success(f"Loaded {len(df)} rows")

            else:
                query = st.sidebar.text_area(
                    "SQL SELECT query",
                    height=140,
                    key="sql_query_input",
                )

                if st.sidebar.button("Run query", key="run_query_btn") and query:
                    df = run_safe_query(
                        st.session_state.db_client,
                        query,
                    )
                    dataset_name = "SQL Query"
                    st.sidebar.success(f"Loaded {len(df)} rows")

    # ---------------- Sample data ----------------
    else:
        st.sidebar.subheader("Sample dataset")
        sample = st.sidebar.selectbox(
            "Choose",
            ["Sales", "Customer", "Random"],
            key="sample_dataset_selector",
        )

        np.random.seed(42)

        if sample == "Sales":
            df = pd.DataFrame(
                {
                    "date": pd.date_range("2023-01-01", periods=365),
                    "revenue": np.random.normal(10_000, 2_000, 365),
                    "units": np.random.poisson(50, 365),
                    "region": np.random.choice(["N", "S", "E", "W"], 365),
                }
            )
            dataset_name = "Sample Sales"

        elif sample == "Customer":
            df = pd.DataFrame(
                {
                    "age": np.random.randint(18, 70, 500),
                    "income": np.random.lognormal(10.5, 0.5, 500),
                    "segment": np.random.choice(["A", "B", "C"], 500),
                }
            )
            dataset_name = "Sample Customer"

        else:
            df = pd.DataFrame(
                {
                    "x": np.random.normal(0, 1, 300),
                    "y": np.random.normal(5, 2, 300),
                    "group": np.random.choice(["A", "B", "C"], 300),
                }
            )
            dataset_name = "Random Data"

        st.sidebar.success(f"Loaded {len(df)} rows")

    # ---------------- Persist ----------------
    if df is not None:
        persist_dataset(df, dataset_name or "Dataset")

    if st.session_state.current_data is not None:
        render_filters_panel()
    else:
        st.session_state[FILTERED_DATA_KEY] = None
        st.session_state[FILTERS_ACTIVE_KEY] = False


# ------------------------------------------------------------------
# Navigation
# ------------------------------------------------------------------
def render_primary_navigation() -> str:
    selected = st.radio(
        "Navigate",
        TAB_TITLES,
        key=NAV_RADIO_KEY,
        horizontal=True,
        label_visibility="collapsed",
    )
    return selected


# ------------------------------------------------------------------
# Filtering
# ------------------------------------------------------------------
def reset_filter_state() -> None:
    """Clear all sidebar filter selections and reset derived state."""

    keys_to_clear = [
        key
        for key in list(st.session_state.keys())
        if key.startswith("filter_cat_")
        or key.startswith("filter_num_")
        or key.startswith("filter_dt_")
    ]

    for key in keys_to_clear:
        st.session_state.pop(key, None)

    st.session_state[FILTERED_DATA_KEY] = st.session_state.current_data
    st.session_state[FILTERS_ACTIVE_KEY] = False
    st.session_state[FILTER_SIGNATURE_KEY] = None
    st.session_state[FILTER_VERSION_KEY] = st.session_state.get(FILTER_VERSION_KEY, 0) + 1


def render_filters_panel() -> None:
    df = st.session_state.current_data
    if df is None or df.empty:
        st.session_state[FILTERED_DATA_KEY] = df
        st.session_state[FILTERS_ACTIVE_KEY] = False
        return

    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")
    if st.sidebar.button("Clear all filters"):
        reset_filter_state()
    filters_active = False
    filtered_df = df
    filter_snapshot: Dict[str, Any] = {}

    categorical_cols = get_categorical_columns(df)
    numeric_cols = [col for col in get_numeric_columns(df) if not pd.api.types.is_bool_dtype(df[col])]
    datetime_cols = get_datetime_columns(df)

    missing_label = "(Missing)"

    if categorical_cols:
        with st.sidebar.expander("Categorical", expanded=False):
            for column in categorical_cols:
                series_source = filtered_df if not filtered_df.empty else df
                series = series_source[column]
                options = sorted(series.dropna().unique().tolist())
                include_missing = series.isna().any()
                display_options = options + ([missing_label] if include_missing else [])
                if not display_options:
                    continue

                state_key = f"filter_cat_{column}"
                default_values = ensure_multiselect_state(state_key, display_options)
                selection = st.sidebar.multiselect(
                    column,
                    display_options,
                    default=default_values,
                    key=state_key,
                )
                filter_snapshot[state_key] = selection.copy()
                option_set = set(display_options)
                selection_set = set(selection)

                if selection_set != option_set:
                    filters_active = True
                    if not selection:
                        filtered_df = filtered_df.iloc[0:0]
                        continue
                    real_values = [val for val in selection if val != missing_label]
                    mask = filtered_df[column].isin(real_values)
                    if include_missing and missing_label in selection:
                        mask |= filtered_df[column].isna()
                    filtered_df = filtered_df[mask]

    if numeric_cols:
        with st.sidebar.expander("Numeric", expanded=False):
            for column in numeric_cols:
                series_source = filtered_df if not filtered_df.empty else df
                series = series_source[column].dropna()
                if series.empty:
                    continue

                is_integer = pd.api.types.is_integer_dtype(series)
                col_min = int(series.min()) if is_integer else float(series.min())
                col_max = int(series.max()) if is_integer else float(series.max())
                if np.isclose(col_min, col_max):
                    continue

                default_range = (col_min, col_max)
                range_key = f"filter_num_{column}"
                sanitized_range = ensure_range_state(range_key, default_range)
                slider_kwargs = {
                    "min_value": col_min,
                    "max_value": col_max,
                    "value": tuple(sanitized_range),
                    "key": range_key,
                }
                if is_integer:
                    slider_kwargs["step"] = 1
                selected_range = st.sidebar.slider(column, **slider_kwargs)
                if is_integer:
                    filter_snapshot[range_key] = [int(selected_range[0]), int(selected_range[1])]
                else:
                    filter_snapshot[range_key] = [float(selected_range[0]), float(selected_range[1])]

                if any(abs(a - b) > 1e-9 for a, b in zip(selected_range, default_range)):
                    filters_active = True
                    lower, upper = selected_range
                    mask = filtered_df[column].between(lower, upper)
                    mask |= filtered_df[column].isna()
                    filtered_df = filtered_df[mask]

    if datetime_cols:
        with st.sidebar.expander("Datetime", expanded=False):
            for column in datetime_cols:
                series_source = filtered_df if not filtered_df.empty else df
                series = pd.to_datetime(series_source[column], errors="coerce").dropna()
                if series.empty:
                    continue

                col_min = series.min().date()
                col_max = series.max().date()
                if col_min > col_max:
                    col_min, col_max = col_max, col_min

                date_key = f"filter_dt_{column}"
                default_dates = (col_min, col_max)
                sanitized_dates = ensure_range_state(date_key, default_dates)
                selected_dates = st.sidebar.date_input(
                    column,
                    value=tuple(sanitized_dates),
                    key=date_key,
                )

                if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
                    start_date, end_date = selected_dates
                else:
                    start_date = end_date = selected_dates

                filter_snapshot[date_key] = [start_date.isoformat(), end_date.isoformat()]

                if start_date and end_date:
                    start_ts = pd.to_datetime(min(start_date, end_date))
                    end_ts = pd.to_datetime(max(start_date, end_date))
                    if (start_ts.date(), end_ts.date()) != default_dates:
                        filters_active = True
                        column_series = pd.to_datetime(filtered_df[column], errors="coerce")
                        mask = column_series.between(start_ts, end_ts, inclusive="both")
                        mask |= column_series.isna()
                        filtered_df = filtered_df[mask]

    st.session_state[FILTERED_DATA_KEY] = filtered_df
    st.session_state[FILTERS_ACTIVE_KEY] = filters_active and len(filtered_df) != len(df)

    signature_payload = {
        key: (list(value) if isinstance(value, (tuple, list)) else value)
        for key, value in filter_snapshot.items()
    }
    signature = json.dumps(signature_payload, sort_keys=True, default=str)
    previous_signature = st.session_state.get(FILTER_SIGNATURE_KEY)
    if previous_signature is None:
        st.session_state[FILTER_SIGNATURE_KEY] = signature
    elif signature != previous_signature:
        st.session_state[FILTER_SIGNATURE_KEY] = signature
        st.session_state[FILTER_VERSION_KEY] += 1


def build_visual_context(
    df: pd.DataFrame,
    *,
    prefix: str,
    numeric_cols: List[str],
    categorical_cols: List[str],
    log_target_columns: Optional[List[str]] = None,
    default_numeric: Optional[str] = None,
) -> VisualFilterContext:
    """Render reusable advanced visual controls behind a shared toggle."""

    log_target_columns = log_target_columns or []
    log_preferences: Dict[str, bool] = {column: False for column in log_target_columns}
    outlier_values: Optional[pd.Series] = None
    outlier_method: Optional[str] = None
    outlier_column: Optional[str] = None
    zscore_threshold: Optional[float] = None
    visual_df = df
    advanced_enabled = False

    missing_label = "(Missing)"
    with st.expander("Visual-level filters", expanded=False):
        toggle_key = f"{prefix}_advanced_toggle"
        st.session_state.setdefault(toggle_key, False)
        advanced_enabled = st.checkbox(
            "Enable visual-level filters (local to this chart)",
            key=toggle_key,
            help="These filters affect only this visualization, not the global dataset.",
        )

        if not advanced_enabled:
            st.caption("Visual-level filters are disabled for this chart.")
        else:
            visual_df = df.copy()

            cat_column = None
            if categorical_cols:
                cat_key = f"{prefix}_cat_col"
                st.session_state.setdefault(cat_key, None)
                if st.session_state[cat_key] not in categorical_cols:
                    st.session_state[cat_key] = None
                cat_column = st.selectbox(
                    "Category filter column",
                    categorical_cols,
                    key=cat_key,
                    index=None,
                    placeholder="Select a column",
                )

            if cat_column:
                series_source = visual_df if not visual_df.empty else df
                series = series_source[cat_column]
                options = sorted(series.dropna().unique().tolist())
                include_missing = series.isna().any()
                display_options = options + ([missing_label] if include_missing else [])
                if display_options:
                    values_key = f"{prefix}_cat_values"
                    default_values = ensure_multiselect_state(values_key, display_options)
                    selected_values = st.multiselect(
                        "Category members",
                        display_options,
                        default=default_values,
                        key=values_key,
                    )
                    if not selected_values:
                        visual_df = visual_df.iloc[0:0]
                    else:
                        chosen = [val for val in selected_values if val != missing_label]
                        mask = visual_df[cat_column].isin(chosen)
                        if include_missing and missing_label in selected_values:
                            mask |= visual_df[cat_column].isna()
                        visual_df = visual_df[mask]

            numeric_column = None
            if numeric_cols:
                numeric_key = f"{prefix}_numeric_col"
                default_value = default_numeric if default_numeric in numeric_cols else None
                st.session_state.setdefault(numeric_key, default_value)
                if st.session_state[numeric_key] not in numeric_cols:
                    st.session_state[numeric_key] = default_value
                numeric_column = st.selectbox(
                    "Numeric range column",
                    numeric_cols,
                    key=numeric_key,
                    index=None,
                    placeholder="Select a column",
                )

            if numeric_column:
                series_source = visual_df if not visual_df.empty else df
                series = series_source[numeric_column].dropna()
                if not series.empty:
                    is_integer = pd.api.types.is_integer_dtype(series)
                    col_min = int(series.min()) if is_integer else float(series.min())
                    col_max = int(series.max()) if is_integer else float(series.max())
                    if not np.isclose(col_min, col_max):
                        range_key = f"{prefix}_numeric_range"
                        sanitized_range = ensure_range_state(range_key, (col_min, col_max))
                        slider_args = {
                            "min_value": col_min,
                            "max_value": col_max,
                            "value": tuple(sanitized_range),
                            "key": range_key,
                        }
                        if is_integer:
                            slider_args["step"] = 1
                        selected_range = st.slider(numeric_column, **slider_args)
                        mask = visual_df[numeric_column].between(selected_range[0], selected_range[1])
                        mask |= visual_df[numeric_column].isna()
                        visual_df = visual_df[mask]

            outlier_column = None
            if numeric_cols:
                outlier_key = f"{prefix}_outlier_col"
                default_value = default_numeric if default_numeric in numeric_cols else None
                st.session_state.setdefault(outlier_key, default_value)
                if st.session_state[outlier_key] not in numeric_cols:
                    st.session_state[outlier_key] = default_value
                outlier_column = st.selectbox(
                    "Outlier column",
                    numeric_cols,
                    key=outlier_key,
                    index=None,
                    placeholder="Select a column",
                )

            method_key = f"{prefix}_outlier_method"
            st.session_state.setdefault(method_key, "None")
            method_choice = st.selectbox(
                "Outlier detection",
                ["None", "IQR", "Z-score"],
                key=method_key,
            )

            exclude_key = f"{prefix}_exclude_outliers"
            st.session_state.setdefault(exclude_key, False)
            exclude_outliers = st.checkbox(
                "Exclude detected outliers",
                key=exclude_key,
            )

            if method_choice != "None" and outlier_column:
                series = visual_df[outlier_column].dropna()
                if not series.empty:
                    if method_choice == "IQR":
                        detected = pd.Series(detect_outliers_iqr(series), name=outlier_column)
                    else:
                        z_key = f"{prefix}_z_thresh"
                        st.session_state.setdefault(z_key, 3.0)
                        zscore_threshold = st.slider(
                            "Z-score threshold",
                            min_value=1.5,
                            max_value=5.0,
                            value=st.session_state[z_key],
                            step=0.1,
                            key=z_key,
                        )
                        detected = detect_outliers_zscore(series, threshold=zscore_threshold)
                    outlier_values = detected
                    outlier_method = method_choice
                    if not detected.empty:
                        st.caption(
                            f"{len(detected)} outliers via {method_choice}. "
                            f"{status_badge('warning', 'Review before interpreting visuals')}"
                        )
                        if exclude_outliers:
                            visual_df = visual_df[~visual_df[outlier_column].isin(detected.values)]

            for column in log_target_columns:
                log_key = f"{prefix}_log_{column}"
                st.session_state.setdefault(log_key, False)
                log_preferences[column] = st.checkbox(
                    f"Use log scale for {column}",
                    key=log_key,
                )

    return VisualFilterContext(
        dataframe=visual_df,
        advanced_enabled=advanced_enabled,
        outlier_column=outlier_column,
        outlier_method=outlier_method,
        outlier_values=outlier_values,
        zscore_threshold=zscore_threshold,
        log_columns=log_preferences,
    )


def detect_geospatial_candidates(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return candidate latitude and longitude columns based on naming heuristics."""

    numeric_cols = get_numeric_columns(df)
    lat_tokens = ("lat", "latitude")
    lon_tokens = ("lon", "lng", "longitude")

    lat_candidates = [
        col
        for col in numeric_cols
        if any(token in col.lower() for token in lat_tokens)
    ]
    lon_candidates = [
        col
        for col in numeric_cols
        if any(token in col.lower() for token in lon_tokens)
    ]

    return lat_candidates, lon_candidates


def coordinates_within_wgs84(latitudes: pd.Series, longitudes: pd.Series) -> bool:
    if latitudes.empty or longitudes.empty:
        return False
    lat_valid = latitudes.between(-90, 90)
    lon_valid = longitudes.between(-180, 180)
    return bool(lat_valid.all() and lon_valid.all())


def is_likely_utm(latitudes: pd.Series, longitudes: pd.Series) -> bool:
    try:
        max_lat = float(np.nanmax(np.abs(latitudes))) if not latitudes.empty else 0.0
        max_lon = float(np.nanmax(np.abs(longitudes))) if not longitudes.empty else 0.0
    except ValueError:
        return False
    return max(max_lat, max_lon) > 1_000_000


def get_cached_projection_series(
    latitudes: pd.Series,
    longitudes: pd.Series,
    epsg_code: int,
) -> Tuple[pd.Series, pd.Series]:
    cache: Dict[Tuple[Any, ...], Tuple[pd.Series, pd.Series]] = st.session_state.setdefault(
        "geo_projection_cache",
        {},
    )
    signature = (
        st.session_state.get(DATASET_VERSION_KEY, 0),
        st.session_state.get(FILTER_VERSION_KEY, 0),
        st.session_state.get(FILTER_SIGNATURE_KEY),
        latitudes.name,
        longitudes.name,
        epsg_code,
        len(latitudes),
    )
    if signature in cache:
        return cache[signature]

    if CRS is None or Transformer is None:
        raise RuntimeError("pyproj is not available in this environment.")

    source_crs = CRS.from_epsg(epsg_code)
    transformer = Transformer.from_crs(source_crs, CRS.from_epsg(4326), always_xy=True)
    lon_vals, lat_vals = transformer.transform(
        longitudes.to_numpy(dtype=float, copy=True),
        latitudes.to_numpy(dtype=float, copy=True),
    )
    projected_lat = pd.Series(lat_vals, index=latitudes.index, name=f"{latitudes.name}_wgs84")
    projected_lon = pd.Series(lon_vals, index=longitudes.index, name=f"{longitudes.name}_wgs84")
    cache[signature] = (projected_lat, projected_lon)
    return cache[signature]


def compute_map_zoom(latitudes: pd.Series, longitudes: pd.Series) -> float:
    lat_span = float(latitudes.max() - latitudes.min()) if not latitudes.empty else 0.0
    lon_span = float(longitudes.max() - longitudes.min()) if not longitudes.empty else 0.0
    span = max(lat_span, lon_span, 0.0001)
    zoom = 8 - np.log10(span)
    return float(np.clip(zoom, 2.0, 12.0))


def build_geospatial_figures(
    data: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    color_col: Optional[str],
    size_col: Optional[str],
    hover_cols: List[str],
    show_density: bool,
    *,
    center: Dict[str, float],
    zoom: float,
    cluster_enabled: bool,
) -> Tuple[go.Figure, Optional[go.Figure]]:
    scatter_kwargs: Dict[str, Any] = {
        "data_frame": data,
        "lat": lat_col,
        "lon": lon_col,
        "hover_data": hover_cols[:6],
        "zoom": zoom,
        "height": 600,
        "center": center,
    }
    if color_col:
        scatter_kwargs["color"] = color_col
    if size_col:
        scatter_kwargs["size"] = size_col

    scatter_fig = px.scatter_mapbox(**scatter_kwargs)
    scatter_fig.update_layout(
        mapbox=dict(style="carto-positron", center=center, zoom=zoom),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    if cluster_enabled:
        scatter_fig.update_traces(cluster=dict(enabled=True))

    density_fig = None
    if show_density:
        density_kwargs: Dict[str, Any] = {
            "data_frame": data,
            "lat": lat_col,
            "lon": lon_col,
            "radius": 20,
            "hover_data": hover_cols[:4],
            "height": 500,
            "center": center,
            "zoom": zoom,
        }
        if size_col:
            density_kwargs["z"] = size_col
        density_fig = px.density_mapbox(**density_kwargs)
        density_fig.update_layout(
            mapbox=dict(style="carto-positron", center=center, zoom=zoom),
            margin=dict(l=0, r=0, t=40, b=0),
        )

    return scatter_fig, density_fig


def render_filtered_dataset_preview() -> None:
    df = get_active_dataframe()
    base_df = st.session_state.current_data
    if df is None or base_df is None or df.empty:
        if df is not None and df.empty:
            st.warning("No records match the current filters.")
        return

    max_rows = min(200, len(df))
    filtered_label = "Filtered dataset preview"
    with st.expander(f"{filtered_label} ({max_rows} rows shown)", expanded=False):
        st.caption(
            f"Displaying {max_rows:,} of {len(df):,} filtered rows (source total: {len(base_df):,})."
        )
        st.dataframe(df.head(max_rows))


# ------------------------------------------------------------------
# Overview tab
# ------------------------------------------------------------------
def render_overview() -> None:
    df = get_active_dataframe()
    profile = get_active_profile()

    if df is None or profile is None:
        st.info("Load a dataset from the sidebar to begin.")
        return

    if df.empty:
        st.warning("No data available after applying filters. Adjust filters to continue.")
        return

    st.header("📊 Dataset Overview")

    if st.button("Recalculate profile", key="recalculate_profile_button"):
        source_df = st.session_state.current_data
        if source_df is not None:
            with st.spinner("Recomputing dataset profile…"):
                updated_profile = cached_profile_dataset(
                    source_df,
                    st.session_state.dataset_name or "Dataset",
                )
            st.session_state.dataset_profile = updated_profile
            profile = updated_profile
            st.success("Profile refreshed.")

    completeness_pct = profile.completeness_score * 100
    mem_mb = profile.memory_usage / 1024 / 1024

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{profile.row_count:,}")
    col2.metric("Columns", profile.column_count)
    col3.metric("Completeness", f"{completeness_pct:.1f}%")
    col4.metric("Memory", f"{mem_mb:.2f} MB")

    st.subheader("Data quality insights")
    for text in InsightGenerator.generate_dataset_overview(profile):
        st.markdown(f"- {text}")

    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)

    focus_col1, focus_col2 = st.columns(2)
    with focus_col1:
        if numeric_cols:
            numeric_focus = stable_selectbox(
                "Numeric column",
                numeric_cols,
                key="overview_numeric_select",
            )
            distribution_options = ["Histogram", "Violin + Box", "ECDF", "Box plot only"]
            distribution_type = stable_selectbox(
                "Distribution type",
                distribution_options,
                key=f"overview_distribution_type_{numeric_focus}",
                default_index=distribution_options.index("Violin + Box"),
            )
            visual_context = build_visual_context(
                df,
                prefix=f"overview_numeric_{numeric_focus}",
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                log_target_columns=[numeric_focus],
                default_numeric=numeric_focus,
            )
            working_df = visual_context.dataframe

            if numeric_focus in working_df.columns:
                plot_df = limit_visual_records(working_df)
                series = plot_df[numeric_focus].dropna()
                if series.empty:
                    st.info("No records match the current visual settings.")
                else:
                    axis_label = numeric_focus
                    outlier_points = (
                        visual_context.outlier_values
                        if visual_context.advanced_enabled
                        else None
                    )
                    log_enabled = visual_context.log_columns.get(numeric_focus, False)
                    if log_enabled:
                        series, removed_count = apply_log_scale_to_series(series)
                        if removed_count:
                            st.warning(
                                f"Log scale excludes {removed_count} non-positive observations."
                            )
                            if outlier_points is not None:
                                outlier_points = outlier_points[outlier_points > 0]
                        if outlier_points is not None and not outlier_points.empty:
                            outlier_points = np.log1p(outlier_points)
                        axis_label = f"log({numeric_focus})"

                    def _build_distribution() -> go.Figure:
                        if distribution_type == "Histogram":
                            return create_histogram_distribution(series, axis_label, outlier_points)
                        if distribution_type == "Violin + Box":
                            return create_violin_distribution(series, axis_label, outlier_points)
                        if distribution_type == "ECDF":
                            return create_ecdf_plot(series, axis_label)
                        if distribution_type == "Box plot only":
                            return create_box_distribution(series, axis_label, outlier_points)
                        raise ValueError(f"Unsupported distribution type: {distribution_type}")

                    histogram_bins = 40 if distribution_type == "Histogram" else None
                    fig = get_cached_distribution_figure(
                        series,
                        axis_label=axis_label,
                        log_enabled=log_enabled,
                        distribution_type=distribution_type,
                        builder=_build_distribution,
                        outlier_points=outlier_points,
                        bins=histogram_bins,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    if (
                        visual_context.advanced_enabled
                        and visual_context.outlier_method == "Z-score"
                        and visual_context.outlier_values is not None
                        and not visual_context.outlier_values.empty
                    ):
                        st.caption(
                            f"Z-score threshold {visual_context.zscore_threshold or 3.0:.1f} → "
                            f"{len(visual_context.outlier_values)} outliers"
                        )
        else:
            st.selectbox(
                "Numeric column",
                ["No numeric columns"],
                key="overview_numeric_select",
            )

    with focus_col2:
        if categorical_cols:
            categorical_focus = stable_selectbox(
                "Categorical column",
                categorical_cols,
                key="overview_categorical_select",
            )
            category_context = build_visual_context(
                df,
                prefix=f"overview_categorical_{categorical_focus}",
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                log_target_columns=[],
                default_numeric=None,
            )
            filtered_cat_df = category_context.dataframe

            if categorical_focus in filtered_cat_df.columns:
                plot_df = limit_visual_records(filtered_cat_df)
                if plot_df.empty:
                    st.info("No records match the current visual settings.")
                else:
                    st.plotly_chart(
                        create_count_plot(plot_df, categorical_focus),
                        use_container_width=True,
                    )
        else:
            st.selectbox(
                "Categorical column",
                ["No categorical columns"],
                key="overview_categorical_select",
            )

    st.subheader("Column health")
    selected_columns = st.multiselect(
        "Columns to inspect",
        df.columns.tolist(),
        default=df.columns.tolist()[: min(5, len(df.columns))],
        key="column_health_select",
    )

    if selected_columns:
        rows = []
        for column in selected_columns:
            col_profile = next((c for c in profile.columns if c.name == column), None)
            if not col_profile:
                continue
            rows.append(
                {
                    "Column": col_profile.name,
                    "Type": col_profile.detected_type,
                    "Null %": f"{col_profile.null_percentage:.1f}",
                    "Unique %": f"{col_profile.unique_percentage:.1f}",
                    "Quality issues": "; ".join(col_profile.quality_issues or []),
                }
            )

        if rows:
            st.dataframe(pd.DataFrame(rows))

    st.subheader("Sample records (filtered)")
    max_preview = min(200, len(df))
    default_preview = min(20, max_preview)
    step = 1 if max_preview < 25 else 5
    sample_rows = st.slider(
        "Rows to preview",
        min_value=1,
        max_value=max_preview,
        value=max(default_preview, 1),
        step=step,
        key="overview_sample_rows",
    )
    st.dataframe(df.head(sample_rows))


# ------------------------------------------------------------------
# Relationships tab
# ------------------------------------------------------------------
def render_relationships() -> None:
    df = get_active_dataframe()
    if df is None:
        st.info("Load a dataset to explore relationships.")
        return

    if df.empty:
        st.warning("No data available after applying filters. Adjust filters to continue.")
        return

    st.header("🔗 Relationships")
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)

    if len(numeric_cols) < 2:
        st.warning("Select a dataset with at least two numeric columns.")
        return

    st.subheader("Correlation explorer")
    default_selection = numeric_cols[: min(3, len(numeric_cols))]

    with st.form("correlation_form"):
        selected_cols = st.multiselect(
            "Numeric columns",
            numeric_cols,
            default=default_selection,
            help="Choose two or more numeric fields to compute correlations.",
            key="correlation_numeric_select",
        )
        corr_method = st.selectbox(
            "Method",
            ["pearson", "spearman", "kendall"],
            index=0,
            key="correlation_method_select",
        )
        bonferroni_toggle = st.checkbox(
            "Apply Bonferroni correction",
            key="bonferroni_toggle",
        )
        run_corr = st.form_submit_button("Analyze correlations")

    if run_corr:
        if len(selected_cols) < 2:
            st.warning("Select at least two numeric columns.")
        else:
            subset_df = df[selected_cols]
            corr_matrix_df = get_cached_correlation(
                subset_df,
                tuple(selected_cols),
                corr_method,
            )
            st.plotly_chart(
                create_correlation_heatmap(corr_matrix_df, title="Correlation matrix"),
                use_container_width=True,
            )

            correlation_results = []
            for col_a, col_b in combinations(selected_cols, 2):
                try:
                    result = calculate_correlation(df, col_a, col_b, method=corr_method)
                    correlation_results.append(result)
                except ValueError as exc:
                    st.warning(f"Skipping {col_a} vs {col_b}: {exc}")

            st.session_state[ANALYSIS_STATE_KEY]["correlations"] = correlation_results

            if correlation_results:
                if bonferroni_toggle:
                    adjusted_alpha = 0.05 / len(correlation_results)
                    st.caption(
                        f"Bonferroni-adjusted alpha: {adjusted_alpha:.4f} {status_badge('warning', 'Conservative threshold')}"
                    )
                else:
                    adjusted_alpha = 0.05

                corr_rows = [
                    {
                        "Variable 1": res.variable1,
                        "Variable 2": res.variable2,
                        "Correlation": round(res.correlation, 4),
                        "p-value": round(res.p_value, 4),
                        "Significant": res.p_value < adjusted_alpha,
                    }
                    for res in correlation_results
                ]
                st.dataframe(pd.DataFrame(corr_rows))

                st.markdown("**Interpretation**")
                for text in InsightGenerator.generate_correlation_insights(correlation_results):
                    st.markdown(f"- {text}")

    with st.expander("Pairwise scatter plot"):
        scatter_cols = [col for col in numeric_cols]
        if scatter_cols:
            x_col = stable_selectbox("X-axis", scatter_cols, key="scatter_x")
            y_col = stable_selectbox(
                "Y-axis",
                scatter_cols,
                key="scatter_y",
                default_index=min(1, len(scatter_cols) - 1),
            )
            color_options = [None] + categorical_cols
            color_col = stable_selectbox(
                "Color by",
                color_options,
                key="scatter_color",
            )
            if x_col and y_col and x_col != y_col:
                scatter_context = build_visual_context(
                    df,
                    prefix="relationships_scatter",
                    numeric_cols=numeric_cols,
                    categorical_cols=categorical_cols,
                    log_target_columns=[x_col, y_col],
                    default_numeric=x_col,
                )
                visual_df = scatter_context.dataframe

                visual_df = visual_df.dropna(subset=[x_col, y_col])
                removed_msgs: List[str] = []
                x_title = x_col
                y_title = y_col
                if scatter_context.log_columns.get(x_col):
                    visual_df, removed = apply_log_scale_to_dataframe(visual_df, x_col)
                    if removed:
                        removed_msgs.append(f"{removed} records removed for log {x_col}")
                    x_title = f"log({x_col})"
                if scatter_context.log_columns.get(y_col):
                    visual_df, removed_y = apply_log_scale_to_dataframe(visual_df, y_col)
                    if removed_y:
                        removed_msgs.append(f"{removed_y} records removed for log {y_col}")
                    y_title = f"log({y_col})"

                if visual_df.empty:
                    st.info("No data available after applying visual filters.")
                else:
                    plot_df = limit_visual_records(visual_df)
                    fig = create_scatter_plot(
                        plot_df,
                        x_col=x_col,
                        y_col=y_col,
                        color_col=color_col,
                        size_col=None,
                        trendline=True,
                    )
                    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
                    st.plotly_chart(fig, use_container_width=True)
                    if removed_msgs:
                        st.caption("; ".join(removed_msgs))

    st.subheader("Regression analysis")
    with st.form("regression_form"):
        dependent_var = stable_selectbox("Dependent variable", numeric_cols, key="regression_dependent")
        available_predictors = [col for col in numeric_cols if col != dependent_var]
        predictor_vars = st.multiselect(
            "Predictor variables",
            available_predictors,
            default=available_predictors[:1],
            key="regression_predictors",
        )
        run_regression = st.form_submit_button("Run regression")

    if run_regression:
        if not predictor_vars:
            st.warning("Select at least one predictor variable.")
        else:
            try:
                regression_context = build_visual_context(
                    df,
                    prefix="regression_block",
                    numeric_cols=numeric_cols,
                    categorical_cols=categorical_cols,
                    log_target_columns=[dependent_var] + predictor_vars,
                    default_numeric=dependent_var,
                )
                working_df = regression_context.dataframe

                regression_df = working_df[[dependent_var] + predictor_vars].dropna()
                if regression_df.empty:
                    st.warning("No rows available after applying visual filters.")
                    raise ValueError("Insufficient data after local filtering")

                axis_titles = {col: col for col in [dependent_var] + predictor_vars}
                log_warnings: List[str] = []
                for column in [dependent_var] + predictor_vars:
                    if regression_context.log_columns.get(column):
                        regression_df, removed = apply_log_scale_to_dataframe(regression_df, column)
                        axis_titles[column] = f"log({column})"
                        if removed:
                            log_warnings.append(f"{removed} records removed for log {column}")

                if regression_df.empty:
                    st.warning("All rows removed by log scaling.")
                    raise ValueError("Log scaling removed all data")

                if log_warnings:
                    st.caption("; ".join(log_warnings))

                regression_result = linear_regression_analysis(regression_df, dependent_var, predictor_vars)
                st.session_state[ANALYSIS_STATE_KEY]["regression"] = {
                    "dependent": dependent_var,
                    "predictors": predictor_vars,
                    "result": regression_result,
                }

                st.markdown("**Model diagnostics**")
                diag_cols = st.columns(4)
                diag_cols[0].metric("R²", f"{regression_result.r_squared:.3f}")
                diag_cols[1].metric("Adj. R²", f"{regression_result.adj_r_squared:.3f}")
                diag_cols[2].metric("MSE", f"{regression_result.mse:.3f}")
                diag_cols[3].metric("RMSE", f"{regression_result.rmse:.3f}")

                st.markdown("**Coefficients**")
                coeff_rows = [
                    {
                        "Variable": var,
                        "Coefficient": round(coef, 4),
                        "p-value": round(regression_result.p_values.get(var, np.nan), 4),
                    }
                    for var, coef in regression_result.coefficients.items()
                ]
                st.dataframe(pd.DataFrame(coeff_rows))

                clean_df = regression_df[[dependent_var] + predictor_vars]
                preds = pd.Series(regression_result.intercept, index=clean_df.index)
                for var in predictor_vars:
                    preds += regression_result.coefficients[var] * clean_df[var]

                plot_df = limit_visual_records(clean_df)
                plot_preds = preds.loc[plot_df.index]

                if len(predictor_vars) == 1:
                    reg_fig = create_regression_plot(
                        plot_df,
                        predictor_vars[0],
                        dependent_var,
                        predictions=plot_preds,
                    )
                    reg_fig.update_layout(
                        xaxis_title=axis_titles.get(predictor_vars[0], predictor_vars[0]),
                        yaxis_title=axis_titles.get(dependent_var, dependent_var),
                    )
                    st.plotly_chart(reg_fig, use_container_width=True)
                else:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=plot_preds,
                            y=plot_df[dependent_var],
                            mode="markers",
                            name="Actual vs predicted",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=plot_preds,
                            y=plot_preds,
                            mode="lines",
                            name="Perfect fit",
                            line=dict(color="green"),
                        )
                    )
                    fig.update_layout(
                        xaxis_title="Predicted",
                        yaxis_title=axis_titles.get(dependent_var, dependent_var),
                        title="Actual vs predicted",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                residuals = clean_df[dependent_var] - preds
                residual_sample = limit_visual_series(residuals)
                st.plotly_chart(
                    create_residual_diagnostic_plot(residual_sample),
                    use_container_width=True,
                )

                significant_predictors = [
                    var for var, p_val in regression_result.p_values.items() if p_val < 0.05
                ]
                if regression_result.r_squared < 0.1 and significant_predictors:
                    st.warning(
                        f"Low explanatory power (R² {regression_result.r_squared:.3f}) despite significant predictors: "
                        f"{', '.join(significant_predictors)}."
                    )

                st.markdown("**Narrative**")
                for text in InsightGenerator.generate_regression_insights(regression_result):
                    st.markdown(f"- {text}")
            except ValueError as exc:
                st.error(str(exc))

    st.subheader("Hypothesis testing")
    with st.expander("Group comparisons"):
        test_options = [
            "Independent t-test",
            "Welch's t-test",
            "Paired t-test",
            "Mann-Whitney U",
            "Wilcoxon signed-rank",
            "One-way ANOVA",
            "Two-way ANOVA",
            "Chi-square test",
            "Levene's test",
        ]
        test_type = st.selectbox("Hypothesis Test Type", test_options, key="hypothesis_test_type")

        def record_test(title: str, insights: List[str]) -> None:
            st.session_state[ANALYSIS_STATE_KEY]["tests"].append({"title": title, "insights": insights})
            st.session_state[ANALYSIS_STATE_KEY]["tests"] = st.session_state[ANALYSIS_STATE_KEY]["tests"][-5:]

        def render_insights(insights: List[str]) -> None:
            if not insights:
                return
            st.markdown("**Narrative**")
            for text in insights:
                st.markdown(f"- {text}")

        if not categorical_cols:
            st.warning("No categorical columns available for grouping.")
        else:
            if test_type in {"Independent t-test", "Welch's t-test", "Mann-Whitney U"}:
                with st.form(f"{test_type.replace(' ', '').lower()}_form"):
                    metric_col = st.selectbox("Numeric metric", numeric_cols, key=f"metric_{test_type}")
                    group_col = st.selectbox("Grouping column", categorical_cols, key=f"group_{test_type}")
                    group_values = sorted(df[group_col].dropna().unique().tolist()) if group_col else []
                    selected_groups = st.multiselect(
                        "Choose two groups",
                        group_values,
                        max_selections=2,
                        key=f"groups_{test_type}",
                    )
                    run_test = st.form_submit_button("Run test")

                if run_test:
                    if len(selected_groups) != 2:
                        st.warning("Select exactly two groups for the test.")
                    else:
                        try:
                            levene_summary = None
                            try:
                                levene_summary = levene_test(
                                    df[df[group_col].isin(selected_groups)],
                                    numeric_col=metric_col,
                                    group_col=group_col,
                                    groups=selected_groups,
                                )
                                variance_badge = status_badge(
                                    "warning" if levene_summary.significant else "significant",
                                    "Unequal variance" if levene_summary.significant else "Equal variance",
                                )
                                st.caption(
                                    f"Levene's test p={levene_summary.p_value:.4f} {variance_badge}"
                                )
                                if levene_summary.significant and test_type == "Independent t-test":
                                    st.info("Recommendation: Use Welch's t-test when variances differ.")
                            except ValueError:
                                pass

                            if test_type == "Independent t-test":
                                test_result = independent_t_test(
                                    df,
                                    numeric_col=metric_col,
                                    group_col=group_col,
                                    group1_value=selected_groups[0],
                                    group2_value=selected_groups[1],
                                )
                                insights = InsightGenerator.generate_ttest_insights(test_result)
                            elif test_type == "Welch's t-test":
                                test_result = welch_t_test(
                                    df,
                                    numeric_col=metric_col,
                                    group_col=group_col,
                                    group1_value=selected_groups[0],
                                    group2_value=selected_groups[1],
                                )
                                insights = InsightGenerator.generate_ttest_insights(test_result)
                            else:
                                rank_result = mann_whitney_u_test(
                                    df,
                                    numeric_col=metric_col,
                                    group_col=group_col,
                                    group1_value=selected_groups[0],
                                    group2_value=selected_groups[1],
                                )
                                insights = InsightGenerator.generate_rank_test_insights(rank_result)

                            st.markdown("**Results**")
                            if test_type == "Mann-Whitney U":
                                summary_df = pd.DataFrame(
                                    [
                                        {
                                            "Statistic": rank_result.statistic_value,
                                            "p-value": rank_result.p_value,
                                            "Effect size": rank_result.effect_size,
                                        }
                                    ]
                                )
                                st.dataframe(summary_df)
                                record_test(f"{test_type}: {metric_col} by {group_col}", insights)
                                render_insights(insights)
                            else:
                                summary_df = pd.DataFrame(
                                    [
                                        {
                                            "Group 1": test_result.group1_name,
                                            "Group 1 mean": test_result.group1_mean,
                                            "Group 2": test_result.group2_name,
                                            "Group 2 mean": test_result.group2_mean,
                                            "Statistic": test_result.t_statistic,
                                            "p-value": test_result.p_value,
                                            "Effect size": test_result.effect_size,
                                        }
                                    ]
                                )
                                st.dataframe(summary_df)
                                record_test(f"{test_type}: {metric_col} by {group_col}", insights)
                                render_insights(insights)

                            st.plotly_chart(
                                create_box_plot_by_group(df.dropna(subset=[metric_col, group_col]), metric_col, group_col),
                                use_container_width=True,
                            )
                        except ValueError as exc:
                            st.error(str(exc))

            elif test_type in {"Paired t-test", "Wilcoxon signed-rank"}:
                with st.form(f"paired_form_{test_type}"):
                    metric_col = st.selectbox("Numeric metric", numeric_cols, key=f"paired_metric_{test_type}")
                    group_col = st.selectbox("Grouping column", categorical_cols, key=f"paired_group_{test_type}")
                    group_values = sorted(df[group_col].dropna().unique().tolist()) if group_col else []
                    selected_groups = st.multiselect(
                        "Choose two paired levels",
                        group_values,
                        max_selections=2,
                        key=f"paired_groups_{test_type}",
                    )
                    available_ids = [col for col in df.columns if col != group_col]
                    pair_id_col = st.selectbox("Pair identifier (subject ID)", available_ids, key=f"pair_id_{test_type}")
                    run_paired = st.form_submit_button("Run paired test")

                if run_paired:
                    if len(selected_groups) != 2:
                        st.warning("Select two levels to compare.")
                    elif not pair_id_col:
                        st.warning("Select a pair identifier column.")
                    else:
                        try:
                            if test_type == "Paired t-test":
                                paired_result = paired_t_test(
                                    df,
                                    numeric_col=metric_col,
                                    group_col=group_col,
                                    pair_id_col=pair_id_col,
                                    group1_value=selected_groups[0],
                                    group2_value=selected_groups[1],
                                )
                                insights = InsightGenerator.generate_ttest_insights(paired_result)
                                record_test(f"Paired t-test: {metric_col}", insights)
                                st.dataframe(
                                    pd.DataFrame(
                                        [
                                            {
                                                "Group 1 mean": paired_result.group1_mean,
                                                "Group 2 mean": paired_result.group2_mean,
                                                "Statistic": paired_result.t_statistic,
                                                "p-value": paired_result.p_value,
                                                "Effect size": paired_result.effect_size,
                                            }
                                        ]
                                    )
                                )
                            else:
                                wilcoxon_result = wilcoxon_signed_rank_test(
                                    df,
                                    numeric_col=metric_col,
                                    group_col=group_col,
                                    pair_id_col=pair_id_col,
                                    group1_value=selected_groups[0],
                                    group2_value=selected_groups[1],
                                )
                                insights = InsightGenerator.generate_rank_test_insights(wilcoxon_result)
                                record_test(f"Wilcoxon: {metric_col}", insights)
                                st.dataframe(
                                    pd.DataFrame(
                                        [
                                            {
                                                "Statistic": wilcoxon_result.statistic_value,
                                                "p-value": wilcoxon_result.p_value,
                                                "Effect size": wilcoxon_result.effect_size,
                                            }
                                        ]
                                    )
                                )

                            render_insights(insights)
                        except ValueError as exc:
                            st.error(str(exc))

            elif test_type == "One-way ANOVA":
                with st.form("anova_form"):
                    metric_col = st.selectbox("Numeric metric", numeric_cols, key="anova_metric")
                    group_col = st.selectbox("Grouping column", categorical_cols, key="anova_group")
                    run_anova = st.form_submit_button("Run ANOVA")

                if run_anova:
                    try:
                        anova_result = anova_test(df, numeric_col=metric_col, group_col=group_col)
                        insights = InsightGenerator.generate_anova_insights(anova_result)
                        record_test(f"ANOVA: {metric_col} by {group_col}", insights)
                        st.markdown("**ANOVA results**")
                        summary_df = pd.DataFrame(
                            [
                                {
                                    "F-statistic": anova_result.f_statistic,
                                    "p-value": anova_result.p_value,
                                    "Significant": anova_result.significant,
                                }
                            ]
                        )
                        st.dataframe(summary_df)
                        st.dataframe(
                            pd.DataFrame(
                                list(anova_result.group_means.items()),
                                columns=["Group", "Mean"],
                            )
                        )
                        render_insights(insights)

                        st.plotly_chart(
                            create_box_plot_by_group(df.dropna(subset=[metric_col, group_col]), metric_col, group_col),
                            use_container_width=True,
                        )

                        if anova_result.significant and len(anova_result.groups) >= 2:
                            st.markdown("**Post-hoc (Tukey's HSD)**")
                            try:
                                tukey_result = tukey_hsd_posthoc(df, numeric_col=metric_col, group_col=group_col)
                                st.dataframe(tukey_result.summary)
                                tukey_insights = InsightGenerator.generate_tukey_insights(tukey_result)
                                record_test(f"Tukey HSD: {group_col}", tukey_insights)
                                render_insights(tukey_insights)
                            except ValueError as exc:
                                st.info(f"Post-hoc unavailable: {exc}")
                    except ValueError as exc:
                        st.error(str(exc))

            elif test_type == "Two-way ANOVA":
                if len(categorical_cols) < 2:
                    st.info("Need at least two categorical columns for two-way ANOVA.")
                else:
                    with st.form("two_way_anova_form"):
                        metric_col = st.selectbox("Numeric metric", numeric_cols, key="two_way_metric")
                        factor_a = st.selectbox("Factor A", categorical_cols, key="factor_a")
                        remaining = [col for col in categorical_cols if col != factor_a]
                        factor_b = st.selectbox("Factor B", remaining, key="factor_b")
                        run_two_way = st.form_submit_button("Run two-way ANOVA")

                    if run_two_way:
                        try:
                            two_way_result = two_way_anova(
                                df,
                                numeric_col=metric_col,
                                factor_a=factor_a,
                                factor_b=factor_b,
                            )
                            insights = InsightGenerator.generate_two_way_anova_insights(two_way_result)
                            record_test(f"Two-way ANOVA: {metric_col}", insights)
                            st.dataframe(two_way_result.anova_table)
                            st.markdown("**Cell means**")
                            st.dataframe(two_way_result.cell_means)
                            render_insights(insights)

                            heatmap_fig = go.Figure(
                                data=go.Heatmap(
                                    z=two_way_result.cell_means.values,
                                    x=two_way_result.cell_means.columns,
                                    y=two_way_result.cell_means.index,
                                    colorscale="Blues",
                                    colorbar=dict(title="Mean"),
                                )
                            )
                            heatmap_fig.update_layout(
                                title=f"Mean {metric_col} by {factor_a} and {factor_b}",
                                xaxis_title=factor_b,
                                yaxis_title=factor_a,
                            )
                            st.plotly_chart(heatmap_fig, use_container_width=True)

                            for factor_name in [(factor_a, "Factor A"), (factor_b, "Factor B")]:
                                label = f"{factor_name[0]} main effect"
                                if (
                                    two_way_result.significant_effects.get(label)
                                    and df[factor_name[0]].nunique() > 2
                                ):
                                    st.markdown(f"**Post-hoc for {factor_name[0]}**")
                                    try:
                                        tukey_result = tukey_hsd_posthoc(df, numeric_col=metric_col, group_col=factor_name[0])
                                        st.dataframe(tukey_result.summary)
                                        tukey_insights = InsightGenerator.generate_tukey_insights(tukey_result)
                                        record_test(f"Tukey HSD: {factor_name[0]}", tukey_insights)
                                        render_insights(tukey_insights)
                                    except ValueError as exc:
                                        st.info(f"Post-hoc unavailable for {factor_name[0]}: {exc}")
                        except ValueError as exc:
                            st.error(str(exc))

            elif test_type == "Chi-square test":
                if len(categorical_cols) < 2:
                    st.info("Need at least two categorical columns for chi-square test.")
                else:
                    with st.form("chi_square_form"):
                        var1 = st.selectbox("Primary categorical variable", categorical_cols, key="chi_var1")
                        remaining = [col for col in categorical_cols if col != var1]
                        var2 = st.selectbox("Comparison categorical variable", remaining, key="chi_var2")
                        run_chi = st.form_submit_button("Run chi-square")

                    if run_chi:
                        try:
                            chi_result = chi_square_test(df, variable1=var1, variable2=var2)
                            insights = InsightGenerator.generate_chi_square_insights(chi_result)
                            record_test(f"Chi-square: {var1} vs {var2}", insights)
                            st.dataframe(chi_result.contingency_table)
                            st.dataframe(chi_result.expected_frequencies)
                            render_insights(insights)

                            heatmap = go.Figure(
                                data=go.Heatmap(
                                    z=chi_result.contingency_table.values,
                                    x=chi_result.contingency_table.columns,
                                    y=chi_result.contingency_table.index,
                                    colorscale="Viridis",
                                    colorbar=dict(title="Count"),
                                )
                            )
                            heatmap.update_layout(
                                title=f"Observed counts: {var1} vs {var2}",
                                xaxis_title=var2,
                                yaxis_title=var1,
                            )
                            st.plotly_chart(heatmap, use_container_width=True)
                        except ValueError as exc:
                            st.error(str(exc))

            elif test_type == "Levene's test":
                with st.form("levene_form"):
                    metric_col = st.selectbox("Numeric metric", numeric_cols, key="levene_metric")
                    group_col = st.selectbox("Grouping column", categorical_cols, key="levene_group")
                    available_groups = sorted(df[group_col].dropna().unique().tolist()) if group_col else []
                    subset_groups = st.multiselect(
                        "Limit to groups (optional)",
                        available_groups,
                        key="levene_groups",
                    )
                    run_levene = st.form_submit_button("Run Levene's test")

                if run_levene:
                    try:
                        levene_result = levene_test(
                            df,
                            numeric_col=metric_col,
                            group_col=group_col,
                            groups=subset_groups or None,
                        )
                        insights = InsightGenerator.generate_levene_insights(levene_result)
                        record_test(f"Levene: {metric_col} by {group_col}", insights)
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {
                                        "F-statistic": levene_result.f_statistic,
                                        "p-value": levene_result.p_value,
                                        "Effect size": levene_result.effect_size,
                                    }
                                ]
                            )
                        )
                        render_insights(insights)
                        st.plotly_chart(
                            create_box_plot_by_group(df.dropna(subset=[metric_col, group_col]), metric_col, group_col),
                            use_container_width=True,
                        )
                    except ValueError as exc:
                        st.error(str(exc))


# ------------------------------------------------------------------
# Trends tab
# ------------------------------------------------------------------
AGG_OPTIONS: Dict[str, str] = {"Mean": "mean", "Sum": "sum", "Median": "median"}
FREQ_OPTIONS: Dict[str, str] = {
    "Daily": "D",
    "Weekly": "W",
    "Monthly": "M",
    "Quarterly": "Q",
}


def aggregate_trend_data(
    df: pd.DataFrame, time_col: str, metric_col: str, agg_func: str, freq_code: Optional[str]
) -> pd.DataFrame:
    subset = df[[time_col, metric_col]].dropna()
    if subset.empty:
        return subset

    if pd.api.types.is_datetime64_any_dtype(subset[time_col]):
        indexed = subset.set_index(time_col).sort_index()
        resampled = indexed[metric_col].resample(freq_code or "D").agg(agg_func)
        aggregated = resampled.reset_index()
    else:
        grouped = subset.sort_values(time_col).groupby(time_col)[metric_col].agg(agg_func)
        aggregated = grouped.reset_index()

    aggregated.columns = [time_col, metric_col]
    return aggregated.dropna(subset=[metric_col])


def build_trend_chart(
    trend_df: pd.DataFrame,
    time_col: str,
    metric_col: str,
    regression_result,
    value_label: Optional[str] = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trend_df[time_col],
            y=trend_df[metric_col],
            mode="lines+markers",
            name="Observed",
        )
    )

    if regression_result:
        idx = np.arange(len(trend_df))
        coef = regression_result.coefficients.get("__trend_index", 0)
        preds = regression_result.intercept + coef * idx
        fig.add_trace(
            go.Scatter(
                x=trend_df[time_col],
                y=preds,
                mode="lines",
                name="Trend line",
                line=dict(color="orange"),
            )
        )

        if len(idx) > 2 and regression_result.mse > 0:
            x_mean = np.mean(idx)
            sxx = np.sum((idx - x_mean) ** 2)
            if sxx > 0:
                t_value = scipy_stats.t.ppf(0.975, df=len(idx) - 2)
                se = np.sqrt(regression_result.mse * (1 / len(idx) + (idx - x_mean) ** 2 / sxx))
                upper = preds + t_value * se
                lower = preds - t_value * se

                fig.add_trace(
                    go.Scatter(
                        x=trend_df[time_col],
                        y=upper,
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=trend_df[time_col],
                        y=lower,
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor="rgba(255, 165, 0, 0.2)",
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    fig.update_layout(
        title="Trend analysis",
        xaxis_title=time_col,
        yaxis_title=value_label or metric_col,
        hovermode="x unified",
    )
    return fig


def render_trends() -> None:
    df = get_active_dataframe()
    if df is None:
        st.info("Load a dataset to evaluate trends.")
        return

    if df.empty:
        st.warning("No data available after applying filters. Adjust filters to continue.")
        return

    st.header("📈 Trends")
    numeric_cols = get_numeric_columns(df)
    time_cols = get_time_columns(df)
    categorical_cols = get_categorical_columns(df)

    if not numeric_cols or not time_cols:
        st.warning("Need at least one numeric and one time/order column.")
        return

    with st.form("trend_form"):
        time_col = stable_selectbox("Time or order column", time_cols, key="trend_time_select")
        metric_col = stable_selectbox(
            "Numeric metric",
            numeric_cols,
            key="trend_metric_select",
            default_index=min(1, len(numeric_cols) - 1),
        )
        agg_label = stable_selectbox("Aggregation", list(AGG_OPTIONS.keys()), key="trend_agg_select")
        freq_label = None
        freq_code = None
        if pd.api.types.is_datetime64_any_dtype(df[time_col]):
            freq_label = stable_selectbox(
                "Resample frequency",
                list(FREQ_OPTIONS.keys()),
                key="trend_freq_select",
                default_index=2,
            )
            freq_code = FREQ_OPTIONS[freq_label]
        run_trend = st.form_submit_button("Analyze trend")

    if run_trend:
        agg_code = AGG_OPTIONS[agg_label]
        trend_visual_context = build_visual_context(
            df,
            prefix="trend_block",
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            log_target_columns=[metric_col],
            default_numeric=metric_col,
        )
        working_df = trend_visual_context.dataframe
        trend_df = aggregate_trend_data(working_df, time_col, metric_col, agg_code, freq_code)
        if trend_df.empty or len(trend_df) < 3:
            st.warning("Not enough data after aggregation. Adjust selections and retry.")
            return

        y_axis_label = metric_col
        if trend_visual_context.log_columns.get(metric_col):
            transformed_series, removed = apply_log_scale_to_series(trend_df[metric_col])
            if transformed_series.empty:
                st.warning("No positive values available for log scaling.")
                return
            trend_df = trend_df.loc[transformed_series.index].copy()
            trend_df[metric_col] = transformed_series
            y_axis_label = f"log({metric_col})"
            if removed:
                st.warning(f"Log scale excludes {removed} aggregated points.")

        trend_series = trend_df[metric_col]
        trend_stats = detect_trend(trend_series)

        trend_df = trend_df.reset_index(drop=True)
        trend_df["__trend_index"] = np.arange(len(trend_df))
        regression_result = linear_regression_analysis(trend_df, metric_col, ["__trend_index"])

        st.session_state[ANALYSIS_STATE_KEY]["trend"] = {
            "time_col": time_col,
            "metric_col": y_axis_label,
            "aggregation": agg_label,
            "frequency": freq_label or "Grouped",
            "stats": trend_stats,
        }

        st.plotly_chart(
            build_trend_chart(trend_df, time_col, metric_col, regression_result, value_label=y_axis_label),
            use_container_width=True,
        )

        kpi_cols = st.columns(4)
        kpi_cols[0].metric("Trend", trend_stats.get("trend", "n/a"))
        kpi_cols[1].metric("Slope", f"{trend_stats.get('slope', 0):.4f}")
        kpi_cols[2].metric("R²", f"{trend_stats.get('r_squared', 0):.3f}")
        kpi_cols[3].metric("p-value", f"{trend_stats.get('p_value', 0):.4f}")

        direction_text = trend_stats.get("trend", "stable")
        narrative = (
            f"{metric_col} aggregated by {agg_label.lower()} over {trend_df[time_col].iloc[0]} to "
            f"{trend_df[time_col].iloc[-1]} shows a {direction_text} pattern."
        )
        st.info(narrative)


# ------------------------------------------------------------------
# Geospatial tab
# ------------------------------------------------------------------
def render_geospatial() -> None:
    df = get_active_dataframe()
    if df is None:
        st.info("Load a dataset with latitude/longitude to explore the map view.")
        return

    if df.empty:
        st.warning("No data available after applying filters. Adjust filters to continue.")
        return

    st.header("🗺️ Geospatial")
    lat_candidates, lon_candidates = detect_geospatial_candidates(df)
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)

    lat_options = lat_candidates or numeric_cols
    lon_options = lon_candidates or numeric_cols

    if not lat_options or not lon_options:
        st.info("Dataset must include numeric latitude and longitude columns.")
        return

    lat_col = stable_selectbox("Latitude column", lat_options, key="geo_lat_col")
    lon_col = stable_selectbox("Longitude column", lon_options, key="geo_lon_col")
    color_col = stable_selectbox("Color by", [None] + categorical_cols, key="geo_color_col")
    size_col = stable_selectbox("Size by", [None] + numeric_cols, key="geo_size_col")

    geo_df = df.dropna(subset=[lat_col, lon_col])
    if geo_df.empty:
        st.warning("No geospatial records available after dropping missing coordinates.")
        return

    lat_series = geo_df[lat_col]
    lon_series = geo_df[lon_col]

    lat_min = float(lat_series.min())
    lat_max = float(lat_series.max())
    lon_min = float(lon_series.min())
    lon_max = float(lon_series.max())

    with st.expander("Coordinate diagnostics", expanded=False):
        bounds_cols = st.columns(2)
        bounds_cols[0].metric("Min latitude", f"{lat_min:.4f}")
        bounds_cols[1].metric("Max latitude", f"{lat_max:.4f}")
        lon_bounds_cols = st.columns(2)
        lon_bounds_cols[0].metric("Min longitude", f"{lon_min:.4f}")
        lon_bounds_cols[1].metric("Max longitude", f"{lon_max:.4f}")
        sample_pairs = pd.DataFrame(
            {
                "Latitude": lat_series.head(5).round(6),
                "Longitude": lon_series.head(5).round(6),
            }
        )
        st.caption("Sample coordinate pairs")
        st.dataframe(sample_pairs, use_container_width=True)

    lat_bounds_valid = coordinates_within_wgs84(lat_series, lon_series)
    if not lat_bounds_valid:
        st.error(
            "Selected columns do not appear to be WGS84 coordinates (degrees). "
            "They may be projected coordinates (e.g., UTM in meters)."
        )
        attempt_projection = st.checkbox(
            "Attempt automatic projection detection",
            key="geo_projection_toggle",
            value=st.session_state.get("geo_projection_toggle", False),
        )
        if not attempt_projection:
            return
        if not is_likely_utm(lat_series, lon_series):
            st.warning("Values exceed degree bounds but do not resemble UTM ranges. Provide WGS84 coordinates or a known EPSG code.")
            return
        epsg_input = st.text_input(
            "Enter source EPSG code (e.g., 32633)",
            value=st.session_state.get("geo_epsg_code", ""),
        )
        if not epsg_input.strip():
            st.info("Provide an EPSG code to attempt conversion.")
            return
        try:
            epsg_code = int(epsg_input)
            st.session_state["geo_epsg_code"] = epsg_input
        except ValueError:
            st.warning("EPSG code must be an integer.")
            return
        try:
            projected_lat, projected_lon = get_cached_projection_series(lat_series, lon_series, epsg_code)
        except RuntimeError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Projection conversion failed: {exc}")
            return
        lat_series = projected_lat
        lon_series = projected_lon
        projection_converted = True
        if not coordinates_within_wgs84(lat_series, lon_series):
            st.warning("Converted coordinates still fall outside valid WGS84 bounds. Verify the EPSG code and retry.")
            return
        st.success(f"Converted EPSG:{epsg_code} coordinates to WGS84.")
        lat_min = float(lat_series.min())
        lat_max = float(lat_series.max())
        lon_min = float(lon_series.min())
        lon_max = float(lon_series.max())

    lat_padding = 0.0005 if lat_min == lat_max else 0.0
    lon_padding = 0.0005 if lon_min == lon_max else 0.0

    with st.expander("Geospatial filters", expanded=False):
        lat_default = (float(lat_min - lat_padding), float(lat_max + lat_padding))
        lon_default = (float(lon_min - lon_padding), float(lon_max + lon_padding))
        lat_state = ensure_range_state("geo_lat_range", lat_default)
        lon_state = ensure_range_state("geo_lon_range", lon_default)
        lat_range = st.slider(
            "Latitude range",
            min_value=float(lat_min - lat_padding),
            max_value=float(lat_max + lat_padding),
            value=lat_state,
            key="geo_lat_range_slider",
        )
        lon_range = st.slider(
            "Longitude range",
            min_value=float(lon_min - lon_padding),
            max_value=float(lon_max + lon_padding),
            value=lon_state,
            key="geo_lon_range_slider",
        )
        st.session_state["geo_lat_range"] = lat_range
        st.session_state["geo_lon_range"] = lon_range
        show_density = st.checkbox("Show density heatmap", key="geo_density_toggle")
        cluster_enabled = st.checkbox("Enable marker clustering", key="geo_cluster_toggle_state")

    mask = lat_series.between(lat_range[0], lat_range[1]) & lon_series.between(lon_range[0], lon_range[1])
    filtered_geo = geo_df.loc[mask]
    filtered_lat = lat_series.loc[filtered_geo.index]
    filtered_lon = lon_series.loc[filtered_geo.index]

    if filtered_geo.empty:
        st.info("No points remain after applying the geospatial filters. Adjust the ranges or projection settings.")
        return

    total_points = len(filtered_geo)
    plot_df = limit_visual_records(filtered_geo)
    lat_plot = filtered_lat.loc[plot_df.index]
    lon_plot = filtered_lon.loc[plot_df.index]
    plot_df = plot_df.copy()
    plot_df["__geo_lat"] = lat_plot
    plot_df["__geo_lon"] = lon_plot

    if len(plot_df) < total_points:
        st.caption(f"{total_points:,} points match the filters (showing a sampled {len(plot_df):,}).")
    else:
        st.caption(f"{total_points:,} points match the filters.")

    hover_cols = [
        col
        for col in plot_df.columns
        if col not in {lat_col, lon_col, "__geo_lat", "__geo_lon"}
    ]

    center = {"lat": float(lat_plot.mean()), "lon": float(lon_plot.mean())}
    zoom = compute_map_zoom(lat_plot, lon_plot)

    scatter_fig, density_fig = build_geospatial_figures(
        plot_df,
        "__geo_lat",
        "__geo_lon",
        color_col,
        size_col,
        hover_cols,
        show_density,
        center=center,
        zoom=zoom,
        cluster_enabled=cluster_enabled,
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

    if density_fig is not None:
        st.plotly_chart(density_fig, use_container_width=True)


# ------------------------------------------------------------------
# Insights tab
# ------------------------------------------------------------------
def render_clipboard_export(text: str, key: str) -> None:
    escaped = json.dumps(text)
    components.html(
        f"""
        <div>
            <button style='padding:8px 12px;margin-bottom:8px' onclick="navigator.clipboard.writeText({escaped});">
                Copy narrative to clipboard
            </button>
        </div>
        """,
        height=60,
    )


def render_insights() -> None:
    df = get_active_dataframe()
    profile = get_active_profile()
    if df is None or profile is None:
        st.info("Load data and run analyses to see insights.")
        return

    if df.empty:
        st.warning("No data available after applying filters. Adjust filters to continue.")
        return

    st.header("🧠 AI Insights")
    analysis_store = st.session_state[ANALYSIS_STATE_KEY]

    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)

    with st.expander("Column focus"):
        focus_numeric = st.multiselect("Numeric columns", numeric_cols, key="insights_numeric_focus")
        focus_categorical = st.multiselect("Categorical columns", categorical_cols, key="insights_categorical_focus")

    if focus_numeric or focus_categorical:
        st.subheader("Column-specific insights")
        for column in focus_numeric + focus_categorical:
            column_profile = next((c for c in profile.columns if c.name == column), None)
            if column_profile:
                st.markdown(f"**{column}**")
                for insight in InsightGenerator.generate_column_insights(column_profile):
                    st.markdown(f"- {insight}")

    correlation_results = analysis_store.get("correlations", [])
    summary_text = InsightGenerator.generate_summary_narrative(profile, correlation_results)
    st.subheader("Narrative summary")
    st.markdown(summary_text)
    render_clipboard_export(summary_text, key="summary_clipboard_button")

    st.subheader("Relationship highlights")
    if correlation_results:
        for text in InsightGenerator.generate_correlation_insights(correlation_results):
            st.markdown(f"- {text}")
    else:
        st.info("Run correlation analysis to populate this section.")

    trend_context = analysis_store.get("trend")
    if trend_context:
        stats = trend_context["stats"]
        st.subheader("Trend insight")
        trend_text = (
            f"{trend_context['metric_col']} ({trend_context['aggregation'].lower()}) over "
            f"{trend_context['time_col']} shows a {stats.get('trend', 'stable')} trend "
            f"with slope {stats.get('slope', 0):.4f} and R² {stats.get('r_squared', 0):.3f}."
        )
        st.markdown(trend_text)
    else:
        st.info("Analyze a trend to unlock time-series insights.")

    regression_context = analysis_store.get("regression")
    if regression_context:
        st.subheader("Regression insight")
        for text in InsightGenerator.generate_regression_insights(regression_context["result"]):
            st.markdown(f"- {text}")

    if analysis_store.get("tests"):
        st.subheader("Hypothesis tests")
        for test in analysis_store["tests"][-5:]:
            st.markdown(f"**{test['title']}**")
            for text in test["insights"]:
                st.markdown(f"- {text}")
    else:
        st.info("Use the Relationships tab to run t-tests or ANOVA.")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    init_session_state()
    render_sidebar()

    selected_tab = render_primary_navigation()
    render_filtered_dataset_preview()

    if selected_tab == "Overview":
        render_overview()
    elif selected_tab == "Relationships":
        render_relationships()
    elif selected_tab == "Trends":
        render_trends()
    elif selected_tab == "Geospatial":
        render_geospatial()
    else:
        render_insights()


if __name__ == "__main__":
    main()
