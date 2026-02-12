"""
Visualization components using Plotly for interactive charts.

Provides reusable, publication-quality visualizations for analytics.
"""

from typing import Optional, List
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats


def create_distribution_plot(
    series: pd.Series,
    plot_type: str = "histogram",
    bins: int = 30,
    *,
    axis_label: Optional[str] = None,
    outlier_points: Optional[pd.Series] = None,
    highlight_color: str = "#d62728",
) -> go.Figure:
    """
    Legacy distribution helper retained for backward compatibility.

    New code paths should prefer the dedicated create_* helpers below.
    """

    _ = highlight_color  # Retained for backward compatibility

    if plot_type == "histogram":
        return create_histogram_distribution(series, axis_label, outlier_points, bins=bins)
    if plot_type == "violin":
        return create_violin_distribution(series, axis_label, outlier_points)
    if plot_type == "box":
        return create_box_distribution(series, axis_label, outlier_points)

    raise ValueError(f"Unknown plot type: {plot_type}")


def _prepare_numeric_series(series: pd.Series) -> pd.Series:
    """Return a float64 series with NaNs removed for plotting."""

    clean_series = series.dropna()
    if clean_series.empty:
        return pd.Series([], dtype="float64")
    return clean_series.astype("float64")


def create_histogram_distribution(
    series: pd.Series,
    axis_label: Optional[str],
    outlier_points: Optional[pd.Series] = None,
    *,
    bins: int = 40,
) -> go.Figure:
    """Histogram + horizontal box overlay with optional outlier markers."""

    axis_label = axis_label or series.name or "Value"
    return create_histogram_box_overlay(
        _prepare_numeric_series(series),
        axis_label=axis_label,
        bins=bins,
        outlier_points=outlier_points,
    )


def create_violin_distribution(
    series: pd.Series,
    axis_label: Optional[str],
    outlier_points: Optional[pd.Series] = None,
) -> go.Figure:
    """Violin + embedded box plot with median guideline and optional outliers."""

    axis_label = axis_label or series.name or "Value"
    clean_series = _prepare_numeric_series(series)
    fig = go.Figure()
    fig.add_trace(
        go.Violin(
            y=clean_series,
            name=axis_label,
            box_visible=True,
            meanline_visible=True,
            points="suspectedoutliers",
            fillcolor="rgba(46,134,222,0.25)",
            line_color="#2E86DE",
            spanmode="soft",
        )
    )

    median_value = float(clean_series.median()) if not clean_series.empty else None
    if median_value is not None and np.isfinite(median_value):
        fig.add_hline(
            y=median_value,
            line_color="#2ECC71",
            line_dash="dash",
            annotation=dict(text="Median", font=dict(color="#2ECC71")),
        )

    if outlier_points is not None:
        outlier_points = outlier_points.dropna()
    if outlier_points is not None and not outlier_points.empty:
        fig.add_trace(
            go.Scatter(
                x=[axis_label] * len(outlier_points),
                y=outlier_points,
                mode="markers",
                name="Outliers",
                marker=dict(color="#E74C3C", symbol="x", size=9),
                hovertemplate="Outlier: %{y:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Violin & Box • {axis_label}",
        yaxis_title=axis_label,
        xaxis_title="",
        showlegend=len(fig.data) > 1,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_ecdf_plot(
    series: pd.Series,
    axis_label: Optional[str],
) -> go.Figure:
    """Render an empirical CDF with cumulative probability on the y-axis."""

    axis_label = axis_label or series.name or "Value"
    clean_series = _prepare_numeric_series(series)
    sorted_values = np.sort(clean_series.values)
    if sorted_values.size == 0:
        sorted_values = np.array([])
    probabilities = np.linspace(1.0 / max(len(sorted_values), 1), 1.0, num=len(sorted_values)) if len(sorted_values) else np.array([])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sorted_values,
            y=probabilities,
            mode="lines",
            name="ECDF",
            line=dict(color="#18A999", width=3),
        )
    )
    fig.update_layout(
        title=f"ECDF • {axis_label}",
        xaxis_title=axis_label,
        yaxis_title="Cumulative probability",
        yaxis=dict(range=[0, 1]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_box_distribution(
    series: pd.Series,
    axis_label: Optional[str],
    outlier_points: Optional[pd.Series] = None,
) -> go.Figure:
    """Vertical box plot with optional highlighted outliers."""

    axis_label = axis_label or series.name or "Value"
    clean_series = _prepare_numeric_series(series)
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=clean_series,
            name=axis_label,
            boxpoints="suspectedoutliers",
            marker=dict(color="#34495E", outliercolor="#E67E22"),
            line=dict(color="#34495E"),
        )
    )

    if outlier_points is not None:
        outlier_points = outlier_points.dropna()
    if outlier_points is not None and not outlier_points.empty:
        fig.add_trace(
            go.Scatter(
                x=[axis_label] * len(outlier_points),
                y=outlier_points,
                mode="markers",
                name="Flagged outliers",
                marker=dict(color="#E74C3C", symbol="x", size=9),
                hovertemplate="Outlier: %{y:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Box Plot • {axis_label}",
        yaxis_title=axis_label,
        showlegend=len(fig.data) > 1,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_histogram_box_overlay(
    series: pd.Series,
    *,
    axis_label: Optional[str] = None,
    bins: int = 40,
    outlier_points: Optional[pd.Series] = None,
) -> go.Figure:
    """Render histogram with synchronized horizontal box plot and optional outliers."""

    clean_series = series.dropna()
    if clean_series.empty:
        clean_series = pd.Series([], dtype="float64")

    axis_label = axis_label or clean_series.name or "Value"
    quartiles = np.percentile(clean_series, [25, 50, 75]) if not clean_series.empty else [np.nan] * 3

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.78, 0.22],
        vertical_spacing=0.04,
    )

    fig.add_trace(
        go.Histogram(
            x=clean_series,
            nbinsx=bins,
            marker=dict(color="#2E86DE", opacity=0.8),
            name="Distribution",
            hovertemplate="%{x:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    if outlier_points is not None and not outlier_points.empty:
        fig.add_trace(
            go.Scatter(
                x=outlier_points,
                y=[0] * len(outlier_points),
                mode="markers",
                name="Outliers",
                marker=dict(color="#E74C3C", symbol="x", size=9),
                hovertemplate="Outlier: %{x:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Box(
            x=clean_series,
            orientation="h",
            name="Quartiles",
            boxpoints=False,
            fillcolor="rgba(255,255,255,0)",
            line=dict(color="#95A5A6"),
        ),
        row=2,
        col=1,
    )

    if clean_series.size:
        q1, median, q3 = quartiles
        for value, style in [
            (q1, dict(color="#F1C40F", dash="dot")),
            (median, dict(color="#2ECC71", width=3)),
            (q3, dict(color="#F1C40F", dash="dot")),
        ]:
            if np.isnan(value):
                continue
            fig.add_vline(
                x=float(value),
                line_color=style.get("color"),
                line_dash=style.get("dash"),
                line_width=style.get("width", 2),
                row=1,
                col=1,
            )
            fig.add_vline(
                x=float(value),
                line_color=style.get("color"),
                line_dash=style.get("dash"),
                line_width=style.get("width", 2),
                row=2,
                col=1,
            )

    fig.update_layout(
        title=f"Distribution of {axis_label}",
        xaxis=dict(title=axis_label, showgrid=True),
        xaxis2=dict(title="", showgrid=False),
        yaxis=dict(title="Count"),
        yaxis2=dict(title="", showticklabels=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=outlier_points is not None and not outlier_points.empty,
    )

    return fig


def create_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Matrix"
) -> go.Figure:
    """
    Create interactive correlation heatmap.
    
    Args:
        corr_matrix: Correlation matrix DataFrame
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale="RdBu",
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="",
        width=700,
        height=600
    )
    
    return fig


def create_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    size_col: Optional[str] = None,
    trendline: bool = False
) -> go.Figure:
    """
    Create scatter plot with optional trendline and grouping.
    
    Args:
        df: DataFrame containing the data
        x_col: Column for x-axis
        y_col: Column for y-axis
        color_col: Column for color grouping (optional)
        size_col: Column for point sizing (optional)
        trendline: Whether to add trendline
        
    Returns:
        Plotly figure object
    """
    trendline_type = "ols" if trendline else None
    
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        trendline=trendline_type,
        title=f"{y_col} vs {x_col}",
        opacity=0.6
    )
    
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        hovermode='closest'
    )
    
    return fig


def create_time_series_plot(
    df: pd.DataFrame,
    date_col: str,
    value_cols: List[str],
    title: str = "Time Series"
) -> go.Figure:
    """
    Create time series line plot.
    
    Args:
        df: DataFrame containing the data
        date_col: Column with datetime values
        value_cols: List of numeric columns to plot
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    for col in value_cols:
        fig.add_trace(go.Scatter(
            x=df[date_col],
            y=df[col],
            name=col,
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=date_col,
        yaxis_title="Value",
        hovermode='x unified'
    )
    
    return fig


def create_bar_chart(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    orientation: str = "v",
    top_n: Optional[int] = None
) -> go.Figure:
    """
    Create bar chart for categorical data.
    
    Args:
        df: DataFrame containing the data
        category_col: Column with categories
        value_col: Column with values
        orientation: v (vertical) or h (horizontal)
        top_n: Show only top N categories
        
    Returns:
        Plotly figure object
    """
    # Aggregate data
    agg_df = df.groupby(category_col)[value_col].sum().sort_values(ascending=False)
    
    if top_n:
        agg_df = agg_df.head(top_n)
    
    if orientation == "h":
        fig = go.Figure(go.Bar(
            x=agg_df.values,
            y=agg_df.index,
            orientation='h'
        ))
        fig.update_layout(
            xaxis_title=value_col,
            yaxis_title=category_col
        )
    else:
        fig = go.Figure(go.Bar(
            x=agg_df.index,
            y=agg_df.values
        ))
        fig.update_layout(
            xaxis_title=category_col,
            yaxis_title=value_col
        )
    
    fig.update_layout(
        title=f"{value_col} by {category_col}",
        showlegend=False
    )
    
    return fig


def create_box_plot_by_group(
    df: pd.DataFrame,
    numeric_col: str,
    group_col: str
) -> go.Figure:
    """
    Create grouped box plot.
    
    Args:
        df: DataFrame containing the data
        numeric_col: Numeric column to analyze
        group_col: Categorical grouping column
        
    Returns:
        Plotly figure object
    """
    fig = px.box(
        df,
        x=group_col,
        y=numeric_col,
        title=f"Distribution of {numeric_col} by {group_col}",
        points="outliers"
    )
    
    fig.update_layout(
        xaxis_title=group_col,
        yaxis_title=numeric_col
    )
    
    return fig


def create_count_plot(
    df: pd.DataFrame,
    column: str,
    top_n: int = 10
) -> go.Figure:
    """
    Create count plot for categorical column.
    
    Args:
        df: DataFrame containing the data
        column: Categorical column to count
        top_n: Show top N categories
        
    Returns:
        Plotly figure object
    """
    value_counts = df[column].value_counts().head(top_n)
    
    fig = go.Figure(go.Bar(
        x=value_counts.index,
        y=value_counts.values
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Values in {column}",
        xaxis_title=column,
        yaxis_title="Count",
        showlegend=False
    )
    
    return fig


def create_regression_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    predictions: Optional[pd.Series] = None
) -> go.Figure:
    """
    Create regression visualization with residuals.
    
    Args:
        df: DataFrame containing the data
        x_col: Independent variable column
        y_col: Dependent variable column
        predictions: Predicted values (optional)
        
    Returns:
        Plotly figure object
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Regression Plot", "Residual Plot")
    )
    
    # Scatter plot with trendline
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode='markers',
            name='Actual',
            opacity=0.6
        ),
        row=1, col=1
    )
    
    if predictions is not None:
        # Add regression line
        sorted_idx = df[x_col].argsort()
        fig.add_trace(
            go.Scatter(
                x=df[x_col].iloc[sorted_idx],
                y=predictions.iloc[sorted_idx],
                mode='lines',
                name='Predicted',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        # Residual plot
        residuals = df[y_col] - predictions
        fig.add_trace(
            go.Scatter(
                x=predictions,
                y=residuals,
                mode='markers',
                name='Residuals',
                opacity=0.6
            ),
            row=1, col=2
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
    
    fig.update_xaxes(title_text=x_col, row=1, col=1)
    fig.update_yaxes(title_text=y_col, row=1, col=1)
    fig.update_xaxes(title_text="Predicted", row=1, col=2)
    fig.update_yaxes(title_text="Residuals", row=1, col=2)
    
    fig.update_layout(
        title="Regression Analysis",
        showlegend=True,
        height=400
    )
    
    return fig


def create_residual_diagnostic_plot(residuals: pd.Series) -> go.Figure:
    """Render residual distribution and Q-Q diagnostics."""

    clean = residuals.dropna()
    if clean.empty:
        clean = pd.Series([0.0], name="residuals")

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Residual Distribution", "Residual Q-Q"))

    fig.add_trace(
        go.Histogram(
            x=clean,
            nbinsx=30,
            marker_color="#1f77b4",
            name="Residuals",
        ),
        row=1,
        col=1,
    )

    sorted_res = np.sort(clean.values)
    probs = (np.arange(1, len(sorted_res) + 1) - 0.5) / len(sorted_res)
    theoretical = scipy_stats.norm.ppf(probs)

    fig.add_trace(
        go.Scatter(
            x=theoretical,
            y=sorted_res,
            mode="markers",
            marker=dict(color="#1f77b4"),
            name="Q-Q",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=theoretical,
            y=theoretical,
            mode="lines",
            line=dict(color="orange", dash="dash"),
            name="Reference",
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Residuals", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Theoretical quantiles", row=1, col=2)
    fig.update_yaxes(title_text="Observed quantiles", row=1, col=2)
    fig.update_layout(showlegend=False, height=400)

    return fig


def create_3d_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    color_col: Optional[str] = None
) -> go.Figure:
    """
    Create 3D scatter plot.
    
    Args:
        df: DataFrame containing the data
        x_col: Column for x-axis
        y_col: Column for y-axis
        z_col: Column for z-axis
        color_col: Column for color grouping (optional)
        
    Returns:
        Plotly figure object
    """
    fig = px.scatter_3d(
        df,
        x=x_col,
        y=y_col,
        z=z_col,
        color=color_col,
        title=f"3D Visualization: {x_col}, {y_col}, {z_col}"
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col
        )
    )
    
    return fig
