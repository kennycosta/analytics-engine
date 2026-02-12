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


def create_distribution_plot(
    df: pd.DataFrame,
    column: str,
    plot_type: str = "histogram",
    bins: int = 30
) -> go.Figure:
    """
    Create distribution visualization for a numeric column.
    
    Args:
        df: DataFrame containing the data
        column: Column name to visualize
        plot_type: histogram, box, or violin
        bins: Number of bins for histogram
        
    Returns:
        Plotly figure object
    """
    if plot_type == "histogram":
        fig = px.histogram(
            df, 
            x=column,
            nbins=bins,
            title=f"Distribution of {column}",
            marginal="box"
        )
        fig.update_layout(
            xaxis_title=column,
            yaxis_title="Count",
            showlegend=False
        )
    
    elif plot_type == "box":
        fig = px.box(
            df,
            y=column,
            title=f"Box Plot of {column}",
            points="outliers"
        )
        fig.update_layout(yaxis_title=column)
    
    elif plot_type == "violin":
        fig = px.violin(
            df,
            y=column,
            title=f"Violin Plot of {column}",
            box=True,
            points="outliers"
        )
        fig.update_layout(yaxis_title=column)
    
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")
    
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
    title: str = "Time Series",
    chart_type: str = "line",
    rolling_window: Optional[int] = None,
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
        series = df[col]
        if rolling_window and rolling_window > 1:
            series = series.rolling(window=rolling_window, min_periods=1).mean()

        trace_kwargs = {
            "x": df[date_col],
            "y": series,
            "name": col,
            "mode": "lines+markers",
        }

        if chart_type == "area":
            trace_kwargs["fill"] = "tozeroy"
            trace_kwargs["mode"] = "lines"
        elif chart_type == "stacked":
            trace_kwargs["stackgroup"] = "one"
            trace_kwargs["mode"] = "lines"

        fig.add_trace(go.Scatter(
            **trace_kwargs
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
