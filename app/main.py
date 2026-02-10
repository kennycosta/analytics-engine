"""
AI Analytics Engine - Streamlit Application

Streamlit UI layer for the analytics platform.
All database access is routed through the db abstraction layer.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Config
from db.connection import DatabaseClient
from db.introspection import get_tables
from db.loader import load_table
from db.query import run_safe_query

from core.profiling import profile_dataset
from core.statistics import (
    correlation_matrix, calculate_correlation, detect_trend
)
from core.insights import InsightGenerator
from core.visualizations import (
    create_distribution_plot,
    create_correlation_heatmap,
    create_scatter_plot
)


# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="AI Analytics Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------
# Session state initialization
# ------------------------------------------------------------------
def init_session_state():
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


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
def render_sidebar():
    st.sidebar.title("🎯 AI Analytics Engine")
    st.sidebar.markdown("---")

    source_type = st.sidebar.radio(
        "Data source",
        ["Upload file", "SQL Database", "Sample data"]
    )

    df = None
    dataset_name = None

    # ---------------- File upload ----------------
    if source_type == "Upload file":
        uploaded = st.sidebar.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "xlsx", "xls"]
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
        server = st.sidebar.text_input("Server", value=cfg.host)
        database = st.sidebar.text_input("Database", value=cfg.database)

        if st.sidebar.button("Connect"):
            try:
                cfg.host = server
                cfg.database = database
                db_client = DatabaseClient(cfg)

                if not db_client.test_connection():
                    raise RuntimeError("Connection test failed")

                st.session_state.db_client = db_client
                st.session_state.available_tables = get_tables(db_client)

                st.sidebar.success("Connected successfully")
            except Exception as exc:
                st.sidebar.error(str(exc))
                st.session_state.db_client = None

        if st.session_state.db_client:
            st.sidebar.markdown("---")
            st.sidebar.subheader("Load data")

            mode = st.sidebar.radio(
                "Load mode",
                ["Table", "Custom SQL (read-only)"]
            )

            row_limit = st.sidebar.number_input(
                "Row limit",
                min_value=100,
                max_value=100_000,
                value=5_000,
                step=500
            )

            if mode == "Table":
                table = st.sidebar.selectbox(
                    "Table",
                    st.session_state.available_tables
                )

                if st.sidebar.button("Load table"):
                    df = load_table(
                        st.session_state.db_client,
                        table_name=table,
                        limit=row_limit
                    )
                    dataset_name = table
                    st.sidebar.success(f"Loaded {len(df)} rows")

            else:
                query = st.sidebar.text_area(
                    "SQL SELECT query",
                    height=140
                )

                if st.sidebar.button("Run query"):
                    df = run_safe_query(
                        st.session_state.db_client,
                        query
                    )
                    dataset_name = "SQL Query"
                    st.sidebar.success(f"Loaded {len(df)} rows")

    # ---------------- Sample data ----------------
    else:
        st.sidebar.subheader("Sample dataset")
        sample = st.sidebar.selectbox(
            "Choose",
            ["Sales", "Customer", "Random"]
        )

        np.random.seed(42)

        if sample == "Sales":
            df = pd.DataFrame({
                "date": pd.date_range("2023-01-01", periods=100),
                "revenue": np.random.normal(10_000, 2_000, 100),
                "units": np.random.poisson(50, 100),
                "region": np.random.choice(["N", "S", "E", "W"], 100)
            })
            dataset_name = "Sample Sales"

        elif sample == "Customer":
            df = pd.DataFrame({
                "age": np.random.randint(18, 70, 200),
                "income": np.random.lognormal(10.5, 0.5, 200),
                "segment": np.random.choice(["A", "B", "C"], 200)
            })
            dataset_name = "Sample Customer"

        else:
            df = pd.DataFrame({
                "x": np.random.normal(0, 1, 150),
                "y": np.random.normal(5, 2, 150),
                "group": np.random.choice(["A", "B", "C"], 150)
            })
            dataset_name = "Random Data"

        st.sidebar.success(f"Loaded {len(df)} rows")

    # ---------------- Persist ----------------
    if df is not None:
        st.session_state.current_data = df
        st.session_state.dataset_name = dataset_name

        with st.spinner("Profiling dataset…"):
            st.session_state.dataset_profile = profile_dataset(df, dataset_name)


# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------
def render_overview():
    if st.session_state.current_data is None:
        st.info("Load data from the sidebar")
        return

    profile = st.session_state.dataset_profile
    st.header("📊 Dataset Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", profile.row_count)
    c2.metric("Columns", profile.column_count)
    c3.metric("Completeness", f"{profile.completeness_score*100:.1f}%")

    st.subheader("Insights")
    for text in InsightGenerator.generate_dataset_overview(profile):
        st.markdown(f"• {text}")

    st.dataframe(st.session_state.current_data.head(20))


def render_relationships():
    df = st.session_state.current_data
    if df is None:
        return

    st.header("🔗 Relationships")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) < 2:
        st.warning("Not enough numeric columns")
        return

    corr = correlation_matrix(df, num_cols)
    st.plotly_chart(
        create_correlation_heatmap(corr),
        use_container_width=True
    )


def render_trends():
    df = st.session_state.current_data
    if df is None:
        return

    st.header("📈 Trends")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    col = st.selectbox("Variable", num_cols)
    result = detect_trend(df[col])

    st.markdown(f"**Trend:** {result['trend']}")
    st.metric("Slope", f"{result.get('slope', 0):.4f}")
    st.metric("R²", f"{result.get('r_squared', 0):.4f}")


def render_insights():
    df = st.session_state.current_data
    profile = st.session_state.dataset_profile

    if df is None:
        return

    st.header("🧠 AI Insights")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corrs = []

    for i, c1 in enumerate(num_cols):
        for c2 in num_cols[i+1:]:
            try:
                corrs.append(calculate_correlation(df, c1, c2))
            except Exception:
                pass

    narrative = InsightGenerator.generate_summary_narrative(profile, corrs)
    st.markdown(narrative)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    init_session_state()
    render_sidebar()

    t1, t2, t3, t4 = st.tabs([
        "Overview",
        "Relationships",
        "Trends",
        "Insights",
    ])

    with t1:
        render_overview()
    with t2:
        render_relationships()
    with t3:
        render_trends()
    with t4:
        render_insights()


if __name__ == "__main__":
    main()
