"""
Statistical analysis engine for correlations, regression, hypothesis testing.

Provides enterprise-grade statistical methods with proper error handling
and result validation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


@dataclass
class CorrelationResult:
    """Result of correlation analysis between two variables."""
    
    variable1: str
    variable2: str
    correlation: float
    p_value: float
    method: str
    significant: bool
    strength: str  # weak, moderate, strong


@dataclass
class RegressionResult:
    """Result of linear regression analysis."""
    
    dependent_var: str
    independent_vars: List[str]
    coefficients: Dict[str, float]
    intercept: float
    r_squared: float
    adj_r_squared: float
    mse: float
    rmse: float
    p_values: Dict[str, float]


@dataclass
class TTestResult:
    """Result of t-test analysis."""
    
    group1_name: str
    group2_name: str
    group1_mean: float
    group2_mean: float
    t_statistic: float
    p_value: float
    significant: bool
    effect_size: float  # Cohen's d


@dataclass
class ANOVAResult:
    """Result of one-way ANOVA test."""
    
    groups: List[str]
    f_statistic: float
    p_value: float
    significant: bool
    group_means: Dict[str, float]


def calculate_correlation(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    method: str = "pearson"
) -> CorrelationResult:
    """
    Calculate correlation between two numeric variables.
    
    Args:
        df: DataFrame containing the data
        col1: First column name
        col2: Second column name
        method: pearson, spearman, or kendall
        
    Returns:
        CorrelationResult with correlation coefficient and significance
    """
    # Remove null values
    clean_df = df[[col1, col2]].dropna()
    
    if len(clean_df) < 3:
        raise ValueError("Insufficient data points for correlation analysis")
    
    # Calculate correlation
    if method == "pearson":
        corr, p_val = stats.pearsonr(clean_df[col1], clean_df[col2])
    elif method == "spearman":
        corr, p_val = stats.spearmanr(clean_df[col1], clean_df[col2])
    elif method == "kendall":
        corr, p_val = stats.kendalltau(clean_df[col1], clean_df[col2])
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Determine significance and strength
    significant = p_val < 0.05
    abs_corr = abs(corr)
    
    if abs_corr < 0.3:
        strength = "weak"
    elif abs_corr < 0.7:
        strength = "moderate"
    else:
        strength = "strong"
    
    return CorrelationResult(
        variable1=col1,
        variable2=col2,
        correlation=corr,
        p_value=p_val,
        method=method,
        significant=significant,
        strength=strength
    )


def correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "pearson"
) -> pd.DataFrame:
    """
    Calculate correlation matrix for numeric columns.
    
    Args:
        df: DataFrame containing the data
        columns: Specific columns to analyze (None = all numeric)
        method: pearson, spearman, or kendall
        
    Returns:
        DataFrame with correlation matrix
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    return df[columns].corr(method=method)


def linear_regression_analysis(
    df: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str]
) -> RegressionResult:
    """
    Perform multiple linear regression analysis.
    
    Args:
        df: DataFrame containing the data
        dependent_var: Target variable name
        independent_vars: List of predictor variable names
        
    Returns:
        RegressionResult with coefficients and metrics
    """
    # Prepare data
    clean_df = df[[dependent_var] + independent_vars].dropna()
    
    if len(clean_df) < len(independent_vars) + 2:
        raise ValueError("Insufficient data points for regression")
    
    X = clean_df[independent_vars]
    y = clean_df[dependent_var]
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predictions
    y_pred = model.predict(X)
    
    # Metrics
    r2 = r2_score(y, y_pred)
    n = len(y)
    k = len(independent_vars)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    
    # Coefficients
    coeffs = dict(zip(independent_vars, model.coef_))
    
    # Calculate p-values
    # Using statsmodels approach manually
    residuals = y - y_pred
    mse_residuals = np.sum(residuals**2) / (n - k - 1)
    var_coef = mse_residuals * np.linalg.inv(X.T @ X).diagonal()
    se_coef = np.sqrt(var_coef)
    t_stats = model.coef_ / se_coef
    p_values = dict(zip(independent_vars, [2 * (1 - stats.t.cdf(abs(t), n - k - 1)) for t in t_stats]))
    
    return RegressionResult(
        dependent_var=dependent_var,
        independent_vars=independent_vars,
        coefficients=coeffs,
        intercept=model.intercept_,
        r_squared=r2,
        adj_r_squared=adj_r2,
        mse=mse,
        rmse=rmse,
        p_values=p_values
    )


def independent_t_test(
    df: pd.DataFrame,
    numeric_col: str,
    group_col: str,
    group1_value: Any,
    group2_value: Any
) -> TTestResult:
    """
    Perform independent samples t-test.
    
    Args:
        df: DataFrame containing the data
        numeric_col: Numeric variable to test
        group_col: Categorical grouping variable
        group1_value: Value identifying group 1
        group2_value: Value identifying group 2
        
    Returns:
        TTestResult with test statistics
    """
    group1_data = df[df[group_col] == group1_value][numeric_col].dropna()
    group2_data = df[df[group_col] == group2_value][numeric_col].dropna()
    
    if len(group1_data) < 2 or len(group2_data) < 2:
        raise ValueError("Insufficient data in one or both groups")
    
    # Perform t-test
    t_stat, p_val = stats.ttest_ind(group1_data, group2_data)
    
    # Calculate Cohen's d (effect size)
    pooled_std = np.sqrt(
        ((len(group1_data) - 1) * group1_data.std()**2 + 
         (len(group2_data) - 1) * group2_data.std()**2) / 
        (len(group1_data) + len(group2_data) - 2)
    )
    cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std
    
    return TTestResult(
        group1_name=str(group1_value),
        group2_name=str(group2_value),
        group1_mean=group1_data.mean(),
        group2_mean=group2_data.mean(),
        t_statistic=t_stat,
        p_value=p_val,
        significant=bool(p_val < 0.05),
        effect_size=cohens_d
    )


def anova_test(
    df: pd.DataFrame,
    numeric_col: str,
    group_col: str
) -> ANOVAResult:
    """
    Perform one-way ANOVA test.
    
    Args:
        df: DataFrame containing the data
        numeric_col: Numeric variable to test
        group_col: Categorical grouping variable
        
    Returns:
        ANOVAResult with F-statistic and group means
    """
    # Get groups
    groups = df[group_col].unique()
    group_data = [df[df[group_col] == g][numeric_col].dropna() for g in groups]
    
    # Filter out groups with insufficient data
    valid_groups = [(g, data) for g, data in zip(groups, group_data) if len(data) >= 2]
    
    if len(valid_groups) < 2:
        raise ValueError("Need at least 2 groups with sufficient data")
    
    groups, group_data = zip(*valid_groups)
    
    # Perform ANOVA
    f_stat, p_val = stats.f_oneway(*group_data)
    
    # Calculate group means
    group_means = {str(g): data.mean() for g, data in zip(groups, group_data)}
    
    return ANOVAResult(
        groups=[str(g) for g in groups],
        f_statistic=f_stat,
        p_value=p_val,
        significant=bool(p_val < 0.05),
        group_means=group_means
    )


def detect_trend(series: pd.Series) -> Dict[str, Any]:
    """
    Detect linear trend in time series or sequential data.
    
    Args:
        series: Pandas series with numeric values
        
    Returns:
        Dictionary with trend statistics
    """
    clean_series = series.dropna()
    
    if len(clean_series) < 3:
        return {"trend": "insufficient_data"}
    
    # Create time index
    x = np.arange(len(clean_series)).reshape(-1, 1)
    y = clean_series.values
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(x, y)
    
    # Determine trend direction and significance
    slope = model.coef_[0]
    y_pred = model.predict(x)
    r2 = r2_score(y, y_pred)
    
    # Statistical test for slope
    residuals = y - y_pred
    mse_residuals = np.sum(residuals**2) / (len(y) - 2)
    se_slope = np.sqrt(mse_residuals / np.sum((x - x.mean())**2))
    t_stat = slope / se_slope
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), len(y) - 2))
    
    if p_val < 0.05:
        if slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
    else:
        trend = "stable"
    
    return {
        "trend": trend,
        "slope": slope,
        "r_squared": r2,
        "p_value": p_val,
        "significant": p_val < 0.05
    }
