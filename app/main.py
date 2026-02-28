"""
AI Analytics Engine - Streamlit Application (v2)

Fully redesigned UI exposing the complete analytics surface:
profiling, column explorer, correlations, statistical tests,
trend detection, and narrative insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Config
from db.connection import DatabaseClient
from db.introspection import get_tables
from db.loader import load_table
from db.query import run_safe_query

from core.profiling import profile_dataset, identify_data_quality_issues
from core.statistics import (
    correlation_matrix,
    calculate_correlation,
    detect_trend,
    linear_regression_analysis,
    independent_t_test,
    anova_test,
)
from core.insights import InsightGenerator
from core.visualizations import (
    create_distribution_plot,
    create_correlation_heatmap,
    create_scatter_plot,
    create_time_series_plot,
    create_box_plot_by_group,
    create_count_plot,
    create_regression_plot,
    create_3d_scatter,
)


# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analytics Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ─────────────────────────────────────────────── */
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0e1117;
    border-right: 1px solid #21262d;
}

/* ── Metric cards ────────────────────────────────────────── */
.kpi-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.5rem;
}
.kpi-card.green  { border-left: 4px solid #3fb950; }
.kpi-card.blue   { border-left: 4px solid #58a6ff; }
.kpi-card.purple { border-left: 4px solid #bc8cff; }
.kpi-card.orange { border-left: 4px solid #f0883e; }
.kpi-card.red    { border-left: 4px solid #f85149; }
.kpi-card.gray   { border-left: 4px solid #30363d; }
.kpi-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8b949e;
    margin-bottom: 0.4rem;
}
.kpi-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #e6edf3;
    line-height: 1.1;
}
.kpi-sub {
    font-size: 0.78rem;
    color: #8b949e;
    margin-top: 0.3rem;
}

/* ── Section header ──────────────────────────────────────── */
.sec-header {
    font-size: 0.67rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #8b949e;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.45rem;
    margin: 1.2rem 0 0.8rem;
}

/* ── Badges ──────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 0.18rem 0.6rem;
    border-radius: 2rem;
    font-size: 0.7rem;
    font-weight: 600;
    line-height: 1.5;
}
.badge.green  { background: #0f2d1b; color: #3fb950; border: 1px solid #1d4a2a; }
.badge.red    { background: #2d0f0f; color: #f85149; border: 1px solid #4a1d1d; }
.badge.blue   { background: #0d2040; color: #58a6ff; border: 1px solid #1d3a5e; }
.badge.purple { background: #1e0f40; color: #bc8cff; border: 1px solid #35205e; }
.badge.orange { background: #2d1800; color: #f0883e; border: 1px solid #4a2d0f; }
.badge.gray   { background: #21262d; color: #8b949e; border: 1px solid #30363d; }

/* ── Alert boxes ─────────────────────────────────────────── */
.alert {
    border-radius: 8px;
    padding: 0.65rem 1rem;
    font-size: 0.84rem;
    margin-bottom: 0.4rem;
    line-height: 1.4;
}
.alert.warning { background: #1c1500; border: 1px solid #9e6a03; color: #e3b341; }
.alert.error   { background: #1a0000; border: 1px solid #da3633; color: #f85149; }
.alert.success { background: #001a0d; border: 1px solid #238636; color: #3fb950; }
.alert.info    { background: #001a33; border: 1px solid #1f6feb; color: #58a6ff; }

/* ── Insight items ───────────────────────────────────────── */
.insight {
    background: #0d1117;
    border-left: 3px solid #58a6ff;
    padding: 0.55rem 0.9rem;
    border-radius: 0 6px 6px 0;
    margin-bottom: 0.45rem;
    font-size: 0.87rem;
    color: #c9d1d9;
    line-height: 1.5;
}

/* ── Stat rows ───────────────────────────────────────────── */
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.4rem 0;
    border-bottom: 1px solid #21262d;
    font-size: 0.84rem;
}
.stat-row:last-child { border-bottom: none; }
.stat-label { color: #8b949e; }
.stat-value { color: #e6edf3; font-weight: 500; font-variant-numeric: tabular-nums; }

/* ── Landing cards ───────────────────────────────────────── */
.feature-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    min-width: 130px;
}
.feature-icon { font-size: 1.8rem; margin-bottom: 0.4rem; }
.feature-label { color: #c9d1d9; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# UI helpers
# ──────────────────────────────────────────────────────────────────────────────

def kpi(label: str, value: str, sub: str = "", color: str = "blue") -> str:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="kpi-card {color}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{sub_html}</div>'
    )


def badge(text: str, color: str = "blue") -> str:
    return f'<span class="badge {color}">{text}</span>'


def alert(text: str, level: str = "warning"):
    st.markdown(f'<div class="alert {level}">{text}</div>', unsafe_allow_html=True)


def insight(text: str):
    st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)


def section(title: str):
    st.markdown(f'<div class="sec-header">{title}</div>', unsafe_allow_html=True)


def stat_row(label: str, value: str):
    st.markdown(
        f'<div class="stat-row">'
        f'<span class="stat-label">{label}</span>'
        f'<span class="stat-value">{value}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def strength_color(strength: str) -> str:
    return {"strong": "green", "moderate": "orange", "weak": "gray"}.get(strength, "gray")


def theme(fig: go.Figure) -> go.Figure:
    """Apply consistent dark theme to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#c9d1d9", size=12),
        title_font=dict(size=13, color="#e6edf3", family="Inter, sans-serif"),
        xaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickcolor="#30363d"),
        yaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickcolor="#30363d"),
        legend=dict(bgcolor="#161b22", bordercolor="#21262d", borderwidth=1),
        margin=dict(t=45, l=10, r=10, b=10),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────────

def init_session_state():
    defaults = {
        "config": None,
        "db_client": None,
        "available_tables": [],
        "current_data": None,
        "dataset_profile": None,
        "dataset_name": None,
        "quality_issues": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if st.session_state.config is None:
        st.session_state.config = Config.load()


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — data loading
# ──────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("## 📊 Analytics Engine")
        st.markdown("---")

        source = st.radio(
            "Source",
            ["Upload file", "SQL Database", "Sample data"],
            label_visibility="collapsed",
        )

        df = None
        dataset_name = None

        # ── File upload ───────────────────────────────────────────────────
        if source == "Upload file":
            st.markdown("**Upload CSV or Excel**")
            uploaded = st.file_uploader(
                "file", type=["csv", "xlsx", "xls"], label_visibility="collapsed"
            )
            if uploaded:
                df = (
                    pd.read_csv(uploaded)
                    if uploaded.name.endswith(".csv")
                    else pd.read_excel(uploaded)
                )
                dataset_name = uploaded.name
                st.success(f"✓ {len(df):,} rows")

        # ── SQL Database ──────────────────────────────────────────────────
        elif source == "SQL Database":
            cfg = st.session_state.config.db
            st.markdown("**Connection**")
            server = st.text_input("Server", value=cfg.host)
            database = st.text_input("Database", value=cfg.database)

            if st.button("Connect", type="primary", use_container_width=True):
                try:
                    cfg.host = server
                    cfg.database = database
                    db_client = DatabaseClient(cfg)
                    if not db_client.test_connection():
                        raise RuntimeError("Connection test failed")
                    st.session_state.db_client = db_client
                    st.session_state.available_tables = get_tables(db_client)
                    st.success("Connected")
                except Exception as exc:
                    st.error(str(exc))
                    st.session_state.db_client = None

            if st.session_state.db_client:
                st.markdown("**Load data**")
                mode = st.radio("Mode", ["Table", "Custom SQL"])
                row_limit = st.number_input("Row limit", 100, 100_000, 5_000, 500)

                if mode == "Table":
                    table = st.selectbox("Table", st.session_state.available_tables)
                    if st.button("Load table", type="primary", use_container_width=True):
                        df = load_table(st.session_state.db_client, table, limit=row_limit)
                        dataset_name = table
                        st.success(f"✓ {len(df):,} rows")
                else:
                    query = st.text_area("SELECT query", height=100)
                    if st.button("Run query", type="primary", use_container_width=True):
                        df = run_safe_query(st.session_state.db_client, query)
                        dataset_name = "SQL Query"
                        st.success(f"✓ {len(df):,} rows")

        # ── Sample data ───────────────────────────────────────────────────
        else:
            sample = st.selectbox("Dataset", ["Sales", "Customer", "Random"])
            np.random.seed(42)

            if sample == "Sales":
                df = pd.DataFrame({
                    "date": pd.date_range("2023-01-01", periods=200),
                    "revenue": np.random.normal(10_000, 2_000, 200),
                    "units": np.random.poisson(50, 200).astype(float),
                    "costs": np.random.normal(6_000, 1_200, 200),
                    "margin": np.random.normal(4_000, 800, 200),
                    "region": np.random.choice(["North", "South", "East", "West"], 200),
                    "channel": np.random.choice(["Online", "Retail", "Partner"], 200),
                })
                dataset_name = "Sample Sales"
            elif sample == "Customer":
                df = pd.DataFrame({
                    "age": np.random.randint(18, 70, 300).astype(float),
                    "income": np.random.lognormal(10.5, 0.5, 300),
                    "spend": np.random.lognormal(8, 0.8, 300),
                    "tenure_months": np.random.randint(1, 60, 300).astype(float),
                    "segment": np.random.choice(["Premium", "Standard", "Basic"], 300),
                    "active": np.random.choice(["Yes", "No"], 300, p=[0.7, 0.3]),
                })
                dataset_name = "Sample Customer"
            else:
                df = pd.DataFrame({
                    "x": np.random.normal(0, 1, 200),
                    "y": np.random.normal(5, 2, 200),
                    "z": np.random.normal(2, 3, 200),
                    "group": np.random.choice(["A", "B", "C"], 200),
                })
                dataset_name = "Random Data"

            st.success(f"✓ {len(df):,} rows")

        # ── Persist & profile ─────────────────────────────────────────────
        if df is not None:
            st.session_state.current_data = df
            st.session_state.dataset_name = dataset_name
            with st.spinner("Profiling dataset…"):
                st.session_state.dataset_profile = profile_dataset(df, dataset_name)
                st.session_state.quality_issues = identify_data_quality_issues(df)

        # Dataset status strip
        if st.session_state.current_data is not None:
            p = st.session_state.dataset_profile
            st.markdown("---")
            st.markdown(f"**{p.name}**")
            st.caption(
                f"{p.row_count:,} rows · {p.column_count} cols · "
                f"{p.memory_usage / 1024:.0f} KB · "
                f"{p.completeness_score*100:.0f}% complete"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Tab 1 — Overview
# ──────────────────────────────────────────────────────────────────────────────

def render_overview():
    if st.session_state.current_data is None:
        st.info("← Load a dataset from the sidebar to begin.")
        return

    df = st.session_state.current_data
    profile = st.session_state.dataset_profile

    # KPI row
    completeness_pct = profile.completeness_score * 100
    c_color = "green" if completeness_pct >= 90 else "orange" if completeness_pct >= 70 else "red"
    mem = profile.memory_usage / 1024
    mem_str = f"{mem:.0f} KB" if mem < 1024 else f"{mem/1024:.1f} MB"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi("Rows", f"{profile.row_count:,}", color="blue"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("Columns", str(profile.column_count), color="purple"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("Completeness", f"{completeness_pct:.1f}%", color=c_color), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi("Memory", mem_str, color="blue"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([3, 2])

    with left:
        section("Column Summary")
        type_colors = {
            "numeric": "#58a6ff", "categorical": "#bc8cff",
            "datetime": "#3fb950", "boolean": "#f0883e",
            "text": "#8b949e", "unknown": "#30363d",
        }
        rows = []
        for cp in profile.columns:
            rows.append({
                "Column": cp.name,
                "Type": cp.detected_type,
                "Non-null": f"{100 - cp.null_percentage:.1f}%",
                "Unique": f"{cp.unique_count:,}",
                "Outliers": len(cp.outliers) if cp.outliers else 0,
                "Issues": len(cp.quality_issues) if cp.quality_issues else 0,
            })
        summary_df = pd.DataFrame(rows)

        def color_type(val):
            return f"color: {type_colors.get(val, '#8b949e')}"

        def color_issues(val):
            if isinstance(val, int) and val > 0:
                return "color: #f0883e; font-weight: 600"
            return ""

        st.dataframe(
            summary_df.style
                .map(color_type, subset=["Type"])
                .map(color_issues, subset=["Issues", "Outliers"]),
            use_container_width=True,
            hide_index=True,
        )

    with right:
        section("Column Types")
        if profile.column_types:
            labels = list(profile.column_types.keys())
            values = list(profile.column_types.values())
            colors = [type_colors.get(l, "#30363d") for l in labels]
            fig = go.Figure(go.Pie(
                labels=labels,
                values=values,
                hole=0.6,
                marker_colors=colors,
                textinfo="label+value",
                hovertemplate="%{label}: %{value}<extra></extra>",
            ))
            theme(fig)
            fig.update_layout(showlegend=False, height=210, margin=dict(t=5, l=5, r=5, b=5))
            st.plotly_chart(fig, use_container_width=True)

        section("Quality Alerts")
        all_issues = list(profile.quality_issues)
        for qi in st.session_state.quality_issues:
            all_issues.append(f"[{qi['column']}] {qi['description']}")

        if all_issues:
            for iss in all_issues:
                alert(f"⚠ {iss}", "warning")
        else:
            alert("✓ No data quality issues detected.", "success")

    st.markdown("<br>", unsafe_allow_html=True)
    section("Data Preview")
    st.dataframe(df.head(50), use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Tab 2 — Column Explorer
# ──────────────────────────────────────────────────────────────────────────────

def render_columns():
    if st.session_state.current_data is None:
        return

    df = st.session_state.current_data
    profile = st.session_state.dataset_profile

    col_name = st.selectbox(
        "Select column", [cp.name for cp in profile.columns], label_visibility="collapsed"
    )
    cp = next(c for c in profile.columns if c.name == col_name)

    type_badge_colors = {
        "numeric": "blue", "categorical": "purple", "datetime": "green",
        "boolean": "orange", "text": "gray",
    }
    st.markdown(
        f"### {cp.name} &nbsp; {badge(cp.detected_type, type_badge_colors.get(cp.detected_type, 'gray'))}",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 2])

    with left:
        section("Profile")
        stat_row("Dtype", cp.dtype)
        stat_row("Detected type", cp.detected_type)
        stat_row("Total values", f"{cp.count:,}")
        stat_row("Non-null", f"{cp.count - cp.null_count:,}")
        stat_row("Null", f"{cp.null_count:,} ({cp.null_percentage:.1f}%)")
        stat_row("Unique", f"{cp.unique_count:,} ({cp.unique_percentage:.1f}%)")

        if cp.numeric_stats:
            ns = cp.numeric_stats
            section("Numeric Stats")
            stat_row("Mean", f"{ns['mean']:,.4f}")
            stat_row("Median", f"{ns['median']:,.4f}")
            stat_row("Std dev", f"{ns['std']:,.4f}")
            stat_row("Min", f"{ns['min']:,.4f}")
            stat_row("Max", f"{ns['max']:,.4f}")
            stat_row("Q25", f"{ns['q25']:,.4f}")
            stat_row("Q75", f"{ns['q75']:,.4f}")
            stat_row("Skewness", f"{ns['skewness']:.4f}")
            stat_row("Kurtosis", f"{ns['kurtosis']:.4f}")
            if cp.outliers:
                stat_row("Outliers", str(len(cp.outliers)))

        if cp.categorical_stats:
            cs = cp.categorical_stats
            section("Categorical Stats")
            stat_row("Mode", str(cs["mode"]))
            mode_pct = (cs["mode_frequency"] / cp.count * 100) if cp.count else 0
            stat_row("Mode frequency", f"{cs['mode_frequency']:,} ({mode_pct:.1f}%)")
            stat_row("Entropy", f"{cs['entropy']:.4f}")

        if cp.datetime_stats:
            ds = cp.datetime_stats
            section("Datetime Stats")
            stat_row("Min date", str(ds["min"])[:10])
            stat_row("Max date", str(ds["max"])[:10])
            stat_row("Range", f"{ds['range_days']} days")

        if cp.quality_issues:
            section("Issues")
            for iss in cp.quality_issues:
                alert(f"⚠ {iss}", "warning")

        section("Insights")
        for ins in InsightGenerator.generate_column_insights(cp):
            insight(ins)

    with right:
        section("Visualization")

        if cp.detected_type == "numeric":
            plot_type = st.radio("Chart type", ["Histogram", "Box", "Violin"], horizontal=True)
            fig = create_distribution_plot(df, col_name, plot_type.lower())
            theme(fig)
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

            if cp.outliers and len(cp.outliers) > 0:
                with st.expander(f"Outlier detail ({len(cp.outliers)} values)"):
                    st.dataframe(
                        pd.DataFrame({"value": cp.outliers}),
                        use_container_width=True,
                        hide_index=True,
                    )

        elif cp.detected_type == "categorical":
            top_n = st.slider("Top N categories", 5, 30, 10)
            fig = create_count_plot(df, col_name, top_n=top_n)
            theme(fig)
            fig.update_traces(marker_color="#58a6ff")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

            section("Value Counts")
            vc = df[col_name].value_counts().reset_index()
            vc.columns = ["Value", "Count"]
            vc["Share %"] = (vc["Count"] / len(df) * 100).round(1).astype(str) + "%"
            st.dataframe(vc.head(top_n), use_container_width=True, hide_index=True)

        elif cp.detected_type == "datetime":
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                value_cols = st.multiselect("Value columns to plot", num_cols, default=num_cols[:1])
                if value_cols:
                    fig = create_time_series_plot(df.sort_values(col_name), col_name, value_cols)
                    theme(fig)
                    fig.update_layout(height=380)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                alert("No numeric columns to plot against this date.", "info")

        else:
            alert("No visualization available for this column type.", "info")


# ──────────────────────────────────────────────────────────────────────────────
# Tab 3 — Relationships
# ──────────────────────────────────────────────────────────────────────────────

def render_relationships():
    if st.session_state.current_data is None:
        return

    df = st.session_state.current_data
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) < 2:
        alert("Need at least 2 numeric columns for relationship analysis.", "info")
        return

    subtab1, subtab2, subtab3 = st.tabs(["Correlation Matrix", "Scatter Explorer", "3D View"])

    # ── Correlation matrix ────────────────────────────────────────────────
    with subtab1:
        method = st.radio("Method", ["pearson", "spearman", "kendall"], horizontal=True)
        corr = correlation_matrix(df, num_cols, method=method)

        fig = create_correlation_heatmap(corr, title=f"{method.capitalize()} Correlation Matrix")
        theme(fig)
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)

        section("Pairwise Correlations")
        results = []
        for i, c1 in enumerate(num_cols):
            for c2 in num_cols[i + 1:]:
                try:
                    results.append(calculate_correlation(df, c1, c2, method=method))
                except Exception:
                    pass

        if results:
            results.sort(key=lambda r: abs(r.correlation), reverse=True)
            rows = [
                {
                    "Variable 1": r.variable1,
                    "Variable 2": r.variable2,
                    "r": round(r.correlation, 4),
                    "p-value": round(r.p_value, 4),
                    "Strength": r.strength,
                    "Significant": "Yes" if r.significant else "No",
                }
                for r in results
            ]
            corr_df = pd.DataFrame(rows)

            def _color_r(val):
                if not isinstance(val, float):
                    return ""
                if abs(val) >= 0.7:
                    return "color: #3fb950; font-weight: 600"
                if abs(val) >= 0.3:
                    return "color: #f0883e"
                return "color: #8b949e"

            def _color_sig(val):
                return "color: #3fb950" if val == "Yes" else "color: #8b949e"

            def _color_strength(val):
                m = {"strong": "color: #3fb950", "moderate": "color: #f0883e", "weak": "color: #8b949e"}
                return m.get(val, "")

            st.dataframe(
                corr_df.style
                    .map(_color_r, subset=["r"])
                    .map(_color_sig, subset=["Significant"])
                    .map(_color_strength, subset=["Strength"]),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            section("Correlation Insights")
            for ins in InsightGenerator.generate_correlation_insights(results):
                insight(ins)

    # ── Scatter explorer ──────────────────────────────────────────────────
    with subtab2:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        c1, c2, c3 = st.columns(3)
        with c1:
            x_col = st.selectbox("X axis", num_cols, key="scat_x")
        with c2:
            y_col = st.selectbox("Y axis", [c for c in num_cols if c != x_col], key="scat_y")
        with c3:
            color_opt = st.selectbox("Color by", ["None"] + cat_cols, key="scat_color")

        trendline = st.toggle("Show trendline", value=True)
        color = color_opt if color_opt != "None" else None

        fig = create_scatter_plot(df, x_col, y_col, color_col=color, trendline=trendline)
        theme(fig)
        fig.update_layout(height=460)
        st.plotly_chart(fig, use_container_width=True)

        try:
            r = calculate_correlation(df, x_col, y_col)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Correlation (r)", f"{r.correlation:.4f}")
            m2.metric("p-value", f"{r.p_value:.4f}")
            m3.metric("Strength", r.strength.capitalize())
            m4.metric("Significant", "Yes" if r.significant else "No")
        except Exception:
            pass

    # ── 3D scatter ────────────────────────────────────────────────────────
    with subtab3:
        if len(num_cols) < 3:
            alert("Need at least 3 numeric columns for 3D visualization.", "info")
        else:
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                x3 = st.selectbox("X axis", num_cols, index=0, key="3d_x")
            with c2:
                y3 = st.selectbox("Y axis", num_cols, index=1, key="3d_y")
            with c3:
                z3 = st.selectbox("Z axis", num_cols, index=2, key="3d_z")
            with c4:
                color3 = st.selectbox("Color by", ["None"] + cat_cols, key="3d_col")

            fig = create_3d_scatter(
                df, x3, y3, z3, color_col=color3 if color3 != "None" else None
            )
            theme(fig)
            fig.update_layout(height=560)
            st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Tab 4 — Statistical Tests
# ──────────────────────────────────────────────────────────────────────────────

def render_tests():
    if st.session_state.current_data is None:
        return

    df = st.session_state.current_data
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    subtab1, subtab2, subtab3 = st.tabs(["Linear Regression", "T-Test", "ANOVA"])

    # ── Linear Regression ─────────────────────────────────────────────────
    with subtab1:
        if len(num_cols) < 2:
            alert("Need at least 2 numeric columns for regression.", "info")
        else:
            cfg_col, res_col = st.columns([1, 2])

            with cfg_col:
                section("Configure Model")
                y_var = st.selectbox("Dependent variable (Y)", num_cols, key="reg_y")
                x_vars = st.multiselect(
                    "Independent variables (X)",
                    [c for c in num_cols if c != y_var],
                    default=[c for c in num_cols if c != y_var][:2],
                    key="reg_x",
                )
                run_btn = st.button("Run Regression", type="primary", use_container_width=True)

            with res_col:
                if run_btn:
                    if not x_vars:
                        alert("Select at least one independent variable.", "warning")
                    else:
                        try:
                            result = linear_regression_analysis(df, y_var, x_vars)

                            section("Model Fit")
                            m1, m2, m3 = st.columns(3)
                            m1.metric("R²", f"{result.r_squared:.4f}")
                            m2.metric("Adj. R²", f"{result.adj_r_squared:.4f}")
                            m3.metric("RMSE", f"{result.rmse:.4f}")

                            section("Coefficients")
                            coef_rows = [
                                {
                                    "Variable": v,
                                    "Coefficient": round(result.coefficients[v], 5),
                                    "p-value": round(result.p_values[v], 4),
                                    "Significant": "Yes" if result.p_values[v] < 0.05 else "No",
                                }
                                for v in x_vars
                            ]
                            coef_df = pd.DataFrame(coef_rows)

                            def _cs(val):
                                return "color: #3fb950" if val == "Yes" else "color: #f85149"

                            st.dataframe(
                                coef_df.style.map(_cs, subset=["Significant"]),
                                use_container_width=True,
                                hide_index=True,
                            )

                            section("Insights")
                            for ins in InsightGenerator.generate_regression_insights(result):
                                insight(ins)

                            if len(x_vars) == 1:
                                section("Regression Plot")
                                from sklearn.linear_model import LinearRegression as _LR
                                clean = df[[x_vars[0], y_var]].dropna()
                                preds = pd.Series(
                                    _LR().fit(clean[[x_vars[0]]], clean[y_var]).predict(clean[[x_vars[0]]]),
                                    index=clean.index,
                                )
                                fig = create_regression_plot(clean, x_vars[0], y_var, preds)
                                theme(fig)
                                st.plotly_chart(fig, use_container_width=True)

                        except Exception as e:
                            alert(f"Regression failed: {e}", "error")

    # ── T-Test ────────────────────────────────────────────────────────────
    with subtab2:
        if not num_cols or not cat_cols:
            alert("Need at least one numeric column and one categorical column.", "info")
        else:
            cfg_col, res_col = st.columns([1, 2])

            with cfg_col:
                section("Configure Test")
                num_col = st.selectbox("Numeric variable", num_cols, key="tt_num")
                grp_col = st.selectbox("Group column", cat_cols, key="tt_grp")
                groups = df[grp_col].dropna().unique().tolist()

                if len(groups) >= 2:
                    g1 = st.selectbox("Group 1", groups, index=0, key="tt_g1")
                    g2 = st.selectbox("Group 2", groups, index=min(1, len(groups) - 1), key="tt_g2")
                    run_tt = st.button("Run T-Test", type="primary", use_container_width=True)
                else:
                    alert("Need at least 2 groups.", "warning")
                    run_tt = False

            with res_col:
                if run_tt and g1 != g2:
                    try:
                        result = independent_t_test(df, num_col, grp_col, g1, g2)

                        section("Group Means")
                        m1, m2, m3 = st.columns(3)
                        m1.metric(f"Mean — {g1}", f"{result.group1_mean:.4f}")
                        m2.metric(f"Mean — {g2}", f"{result.group2_mean:.4f}")
                        m3.metric("Difference", f"{abs(result.group1_mean - result.group2_mean):.4f}")

                        section("Test Results")
                        m4, m5, m6, m7 = st.columns(4)
                        m4.metric("t-statistic", f"{result.t_statistic:.4f}")
                        m5.metric("p-value", f"{result.p_value:.4f}")
                        m6.metric("Significant", "Yes" if result.significant else "No")
                        m7.metric("Cohen's d", f"{result.effect_size:.4f}")

                        section("Insights")
                        for ins in InsightGenerator.generate_ttest_insights(result):
                            insight(ins)

                        section("Group Distributions")
                        subset = df[df[grp_col].isin([g1, g2])]
                        fig = create_box_plot_by_group(subset, num_col, grp_col)
                        theme(fig)
                        fig.update_layout(height=360)
                        st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        alert(f"T-test failed: {e}", "error")

    # ── ANOVA ─────────────────────────────────────────────────────────────
    with subtab3:
        if not num_cols or not cat_cols:
            alert("Need at least one numeric column and one categorical column.", "info")
        else:
            cfg_col, res_col = st.columns([1, 2])

            with cfg_col:
                section("Configure Test")
                num_col_a = st.selectbox("Numeric variable", num_cols, key="anova_num")
                grp_col_a = st.selectbox("Group column", cat_cols, key="anova_grp")
                run_anova = st.button("Run ANOVA", type="primary", use_container_width=True)

            with res_col:
                if run_anova:
                    try:
                        result = anova_test(df, num_col_a, grp_col_a)

                        section("Test Results")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("F-statistic", f"{result.f_statistic:.4f}")
                        m2.metric("p-value", f"{result.p_value:.4f}")
                        m3.metric("Significant", "Yes" if result.significant else "No")

                        section("Group Means")
                        means_df = (
                            pd.DataFrame(list(result.group_means.items()), columns=["Group", "Mean"])
                            .sort_values("Mean", ascending=False)
                        )
                        fig = go.Figure(go.Bar(
                            x=means_df["Group"],
                            y=means_df["Mean"],
                            marker_color="#58a6ff",
                            text=[f"{v:.2f}" for v in means_df["Mean"]],
                            textposition="outside",
                        ))
                        theme(fig)
                        fig.update_layout(
                            title=f"Mean {num_col_a} by {grp_col_a}",
                            height=320,
                            yaxis_title=num_col_a,
                            xaxis_title=grp_col_a,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        section("Box Plot by Group")
                        fig2 = create_box_plot_by_group(df, num_col_a, grp_col_a)
                        theme(fig2)
                        fig2.update_layout(height=340)
                        st.plotly_chart(fig2, use_container_width=True)

                        section("Insights")
                        for ins in InsightGenerator.generate_anova_insights(result):
                            insight(ins)

                    except Exception as e:
                        alert(f"ANOVA failed: {e}", "error")


# ──────────────────────────────────────────────────────────────────────────────
# Tab 5 — Trends
# ──────────────────────────────────────────────────────────────────────────────

def render_trends():
    if st.session_state.current_data is None:
        return

    df = st.session_state.current_data
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    if not num_cols:
        alert("No numeric columns available.", "info")
        return

    left, right = st.columns([1, 2])

    with left:
        section("Select Variable")
        col = st.selectbox("Column", num_cols, label_visibility="collapsed")
        result = detect_trend(df[col])
        trend = result.get("trend", "unknown")

        trend_colors = {
            "increasing": "green", "decreasing": "red",
            "stable": "blue", "insufficient_data": "gray",
        }
        trend_icons = {"increasing": "↑", "decreasing": "↓", "stable": "→"}

        st.markdown(
            kpi(
                "Trend Direction",
                f"{trend_icons.get(trend, '?')} {trend.replace('_', ' ').capitalize()}",
                color=trend_colors.get(trend, "gray"),
            ),
            unsafe_allow_html=True,
        )
        if result.get("slope") is not None:
            st.markdown(kpi("Slope", f"{result['slope']:.5f}", color="blue"), unsafe_allow_html=True)
            st.markdown(kpi("R²", f"{result['r_squared']:.4f}", color="purple"), unsafe_allow_html=True)
            st.markdown(kpi("p-value", f"{result['p_value']:.4f}", color="orange"), unsafe_allow_html=True)
            sig_color = "green" if result.get("significant") else "gray"
            st.markdown(
                kpi("Significant", "Yes" if result.get("significant") else "No", color=sig_color),
                unsafe_allow_html=True,
            )

        section("All Column Trends")
        trend_rows = []
        for c in num_cols:
            tr = detect_trend(df[c])
            trend_rows.append({
                "Column": c,
                "Trend": tr.get("trend", "—"),
                "Slope": round(tr.get("slope", 0) or 0, 5),
                "R²": round(tr.get("r_squared", 0) or 0, 4),
                "Significant": "Yes" if tr.get("significant") else "No",
            })

        def _color_trend(val):
            return {
                "increasing": "color: #3fb950", "decreasing": "color: #f85149",
                "stable": "color: #58a6ff",
            }.get(val, "color: #8b949e")

        def _color_sig(val):
            return "color: #3fb950" if val == "Yes" else "color: #8b949e"

        st.dataframe(
            pd.DataFrame(trend_rows)
                .style.map(_color_trend, subset=["Trend"])
                .map(_color_sig, subset=["Significant"]),
            use_container_width=True,
            hide_index=True,
        )

    with right:
        section("Visualization")

        if date_cols:
            date_col = st.selectbox("Date column", date_cols)
            value_cols = st.multiselect("Value columns", num_cols, default=[col])
            if value_cols:
                sorted_df = df.sort_values(date_col)
                fig = create_time_series_plot(sorted_df, date_col, value_cols)
                theme(fig)
                fig.update_layout(height=420, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Sequential index trend plot
            series = df[col].dropna().reset_index(drop=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=col,
                line=dict(color="#58a6ff", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(88, 166, 255, 0.06)",
            ))
            if result.get("slope") is not None:
                n = len(series)
                intercept = series.mean() - result["slope"] * (n - 1) / 2
                trend_y = result["slope"] * np.arange(n) + intercept
                fig.add_trace(go.Scatter(
                    x=np.arange(n),
                    y=trend_y,
                    mode="lines",
                    name="Trend",
                    line=dict(color="#f0883e", width=2, dash="dash"),
                ))
            theme(fig)
            fig.update_layout(
                title=f"{col} — Sequential Values with Trend",
                height=420,
                xaxis_title="Index",
                yaxis_title=col,
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Tab 6 — Insights
# ──────────────────────────────────────────────────────────────────────────────

def render_insights():
    if st.session_state.current_data is None:
        return

    df = st.session_state.current_data
    profile = st.session_state.dataset_profile
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    corrs = []
    for i, c1 in enumerate(num_cols):
        for c2 in num_cols[i + 1:]:
            try:
                corrs.append(calculate_correlation(df, c1, c2))
            except Exception:
                pass

    left, right = st.columns([1, 1])

    with left:
        section("Executive Summary")
        narrative = InsightGenerator.generate_summary_narrative(profile, corrs)
        for para in narrative.split("\n\n"):
            insight(para)

        st.markdown("<br>", unsafe_allow_html=True)
        section("Dataset Overview")
        for ins in InsightGenerator.generate_dataset_overview(profile):
            insight(ins)

        st.markdown("<br>", unsafe_allow_html=True)
        section("Correlation Findings")
        for ins in InsightGenerator.generate_correlation_insights(corrs):
            insight(ins)

    with right:
        section("Column Insights")
        for cp in profile.columns:
            col_insights = InsightGenerator.generate_column_insights(cp)
            with st.expander(f"{cp.name}  ·  {cp.detected_type}"):
                for ins in col_insights:
                    insight(ins)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    init_session_state()
    render_sidebar()

    if st.session_state.current_data is None:
        st.markdown("""
        <div style="text-align:center;padding:5rem 2rem;">
            <div style="font-size:3.5rem;margin-bottom:1rem">📊</div>
            <h1 style="color:#e6edf3;font-size:2rem;font-weight:700;margin:0 0 0.6rem">
                Analytics Engine
            </h1>
            <p style="color:#8b949e;font-size:1rem;max-width:480px;margin:0 auto 2.5rem;line-height:1.6">
                Load a dataset from the sidebar — upload a CSV, connect to a SQL database,
                or explore one of the built-in sample datasets.
            </p>
            <div style="display:flex;gap:0.9rem;justify-content:center;flex-wrap:wrap">
                <div class="feature-card"><div class="feature-icon">🔍</div><div class="feature-label">Column Profiling</div></div>
                <div class="feature-card"><div class="feature-icon">🔗</div><div class="feature-label">Correlations</div></div>
                <div class="feature-card"><div class="feature-icon">🧪</div><div class="feature-label">Stat Tests</div></div>
                <div class="feature-card"><div class="feature-icon">📈</div><div class="feature-label">Trend Detection</div></div>
                <div class="feature-card"><div class="feature-icon">🧠</div><div class="feature-label">AI Insights</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📊  Overview",
        "🔍  Columns",
        "🔗  Relationships",
        "🧪  Statistical Tests",
        "📈  Trends",
        "🧠  Insights",
    ])

    with t1:
        render_overview()
    with t2:
        render_columns()
    with t3:
        render_relationships()
    with t4:
        render_tests()
    with t5:
        render_trends()
    with t6:
        render_insights()


if __name__ == "__main__":
    main()
