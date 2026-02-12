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
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd


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
    effect_size: float  # Effect size (e.g., Cohen's d)
    test_name: str = "Independent t-test"


@dataclass
class ANOVAResult:
    """Result of one-way ANOVA test."""
    
    groups: List[str]
    f_statistic: float
    p_value: float
    significant: bool
    group_means: Dict[str, float]


@dataclass
class RankTestResult:
    """Result of non-parametric rank-based tests (e.g., Mann-Whitney, Wilcoxon)."""

    test_name: str
    statistic_name: str
    statistic_value: float
    group1_name: str
    group2_name: str
    p_value: float
    effect_size: float
    significant: bool


@dataclass
class ChiSquareResult:
    """Result of chi-square test of independence."""

    variable1: str
    variable2: str
    chi_square: float
    p_value: float
    dof: int
    significant: bool
    cramers_v: float
    contingency_table: pd.DataFrame
    expected_frequencies: pd.DataFrame


@dataclass
class TwoWayANOVAResult:
    """Result of two-way ANOVA including interaction effects."""

    dependent_var: str
    factor_a: str
    factor_b: str
    anova_table: pd.DataFrame
    effect_sizes: Dict[str, float]
    significant_effects: Dict[str, bool]
    cell_means: pd.DataFrame


@dataclass
class TukeyHSDResult:
    """Result of Tukey's HSD post-hoc comparisons."""

    group_col: str
    summary: pd.DataFrame
    significant_pairs: List[Tuple[str, str]]


@dataclass
class LeveneResult:
    """Result of Levene's test for equality of variances."""

    group_col: str
    numeric_col: str
    f_statistic: float
    p_value: float
    significant: bool
    effect_size: Optional[float]


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
        effect_size=cohens_d,
        test_name="Independent t-test"
    )


def welch_t_test(
    df: pd.DataFrame,
    numeric_col: str,
    group_col: str,
    group1_value: Any,
    group2_value: Any
) -> TTestResult:
    """Perform Welch's t-test for unequal variances."""

    group1_data = df[df[group_col] == group1_value][numeric_col].dropna()
    group2_data = df[df[group_col] == group2_value][numeric_col].dropna()

    if len(group1_data) < 2 or len(group2_data) < 2:
        raise ValueError("Insufficient data in one or both groups for Welch's t-test")

    t_stat, p_val = stats.ttest_ind(group1_data, group2_data, equal_var=False)

    sd1 = group1_data.std(ddof=1)
    sd2 = group2_data.std(ddof=1)
    pooled = np.sqrt((sd1**2 + sd2**2) / 2) if np.isfinite(sd1) and np.isfinite(sd2) else np.nan
    if np.isnan(pooled) or np.isclose(pooled, 0):
        effect_size = np.nan
    else:
        effect_size = (group1_data.mean() - group2_data.mean()) / pooled

    return TTestResult(
        group1_name=str(group1_value),
        group2_name=str(group2_value),
        group1_mean=group1_data.mean(),
        group2_mean=group2_data.mean(),
        t_statistic=t_stat,
        p_value=p_val,
        significant=bool(p_val < 0.05),
        effect_size=effect_size,
        test_name="Welch's t-test"
    )


def paired_t_test(
    df: pd.DataFrame,
    numeric_col: str,
    group_col: str,
    pair_id_col: str,
    group1_value: Any,
    group2_value: Any
) -> TTestResult:
    """Perform paired samples t-test using a subject identifier for pairing."""

    if pair_id_col not in df.columns:
        raise ValueError("Pair identifier column is required for paired tests")

    subset = df[[numeric_col, group_col, pair_id_col]].dropna()
    if subset.empty:
        raise ValueError("No complete rows available for paired analysis")

    pivoted = subset.pivot_table(index=pair_id_col, columns=group_col, values=numeric_col)
    missing_cols = [val for val in (group1_value, group2_value) if val not in pivoted.columns]
    if missing_cols:
        raise ValueError(f"Missing data for groups: {', '.join(map(str, missing_cols))}")

    paired_values = pivoted[[group1_value, group2_value]].dropna()
    if len(paired_values) < 2:
        raise ValueError("Need at least two paired observations to run the test")

    data1 = paired_values[group1_value]
    data2 = paired_values[group2_value]
    t_stat, p_val = stats.ttest_rel(data1, data2)

    diffs = data1 - data2
    diff_std = diffs.std(ddof=1)
    if np.isnan(diff_std) or np.isclose(diff_std, 0):
        effect_size = np.nan
    else:
        effect_size = diffs.mean() / diff_std

    return TTestResult(
        group1_name=str(group1_value),
        group2_name=str(group2_value),
        group1_mean=data1.mean(),
        group2_mean=data2.mean(),
        t_statistic=t_stat,
        p_value=p_val,
        significant=bool(p_val < 0.05),
        effect_size=effect_size,
        test_name="Paired t-test"
    )


def mann_whitney_u_test(
    df: pd.DataFrame,
    numeric_col: str,
    group_col: str,
    group1_value: Any,
    group2_value: Any
) -> RankTestResult:
    """Perform Mann-Whitney U test for independent samples."""

    group1_data = df[df[group_col] == group1_value][numeric_col].dropna()
    group2_data = df[df[group_col] == group2_value][numeric_col].dropna()

    if len(group1_data) < 2 or len(group2_data) < 2:
        raise ValueError("Insufficient data in one or both groups for Mann-Whitney U test")

    u_stat, p_val = stats.mannwhitneyu(group1_data, group2_data, alternative="two-sided")

    n1, n2 = len(group1_data), len(group2_data)
    mean_u = n1 * n2 / 2
    std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z_score = (u_stat - mean_u) / std_u if std_u > 0 else np.nan
    effect_size = abs(z_score) / np.sqrt(n1 + n2) if not np.isnan(z_score) else np.nan

    return RankTestResult(
        test_name="Mann-Whitney U",
        statistic_name="U",
        statistic_value=u_stat,
        group1_name=str(group1_value),
        group2_name=str(group2_value),
        p_value=p_val,
        effect_size=effect_size,
        significant=bool(p_val < 0.05)
    )


def wilcoxon_signed_rank_test(
    df: pd.DataFrame,
    numeric_col: str,
    group_col: str,
    pair_id_col: str,
    group1_value: Any,
    group2_value: Any
) -> RankTestResult:
    """Perform Wilcoxon signed-rank test for paired samples."""

    if pair_id_col not in df.columns:
        raise ValueError("Pair identifier column is required for Wilcoxon test")

    subset = df[[numeric_col, group_col, pair_id_col]].dropna()
    pivoted = subset.pivot_table(index=pair_id_col, columns=group_col, values=numeric_col)
    missing_cols = [val for val in (group1_value, group2_value) if val not in pivoted.columns]
    if missing_cols:
        raise ValueError(f"Missing data for groups: {', '.join(map(str, missing_cols))}")

    paired_values = pivoted[[group1_value, group2_value]].dropna()
    if len(paired_values) < 2:
        raise ValueError("Need at least two paired observations to run the test")

    data1 = paired_values[group1_value]
    data2 = paired_values[group2_value]

    stat, p_val = stats.wilcoxon(data1, data2, zero_method="wilcox", alternative="two-sided")

    n = len(paired_values)
    mean_w = n * (n + 1) / 4
    std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z_score = (stat - mean_w) / std_w if std_w > 0 else np.nan
    effect_size = abs(z_score) / np.sqrt(n) if not np.isnan(z_score) else np.nan

    return RankTestResult(
        test_name="Wilcoxon signed-rank",
        statistic_name="W",
        statistic_value=stat,
        group1_name=str(group1_value),
        group2_name=str(group2_value),
        p_value=p_val,
        effect_size=effect_size,
        significant=bool(p_val < 0.05)
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


def two_way_anova(
    df: pd.DataFrame,
    numeric_col: str,
    factor_a: str,
    factor_b: str
) -> TwoWayANOVAResult:
    """Perform two-way ANOVA with interaction between two categorical factors."""

    subset = df[[numeric_col, factor_a, factor_b]].dropna()
    if subset.empty:
        raise ValueError("No data available for two-way ANOVA")

    if subset[factor_a].nunique() < 2 or subset[factor_b].nunique() < 2:
        raise ValueError("Each factor must have at least two levels")

    formula = f"{numeric_col} ~ C({factor_a}) * C({factor_b})"
    model = ols(formula, data=subset).fit()
    anova_res = anova_lm(model, typ=2)

    if "Residual" not in anova_res.index:
        raise ValueError("Could not compute residuals for ANOVA")

    ss_error = anova_res.loc["Residual", "sum_sq"]
    effect_map = {
        f"C({factor_a})": f"{factor_a} main effect",
        f"C({factor_b})": f"{factor_b} main effect",
        f"C({factor_a}):C({factor_b})": f"{factor_a} × {factor_b} interaction",
    }

    effect_sizes: Dict[str, float] = {}
    significant_effects: Dict[str, bool] = {}
    for raw_name, label in effect_map.items():
        if raw_name in anova_res.index:
            ss_effect = anova_res.loc[raw_name, "sum_sq"]
            denom = ss_effect + ss_error
            effect_sizes[label] = ss_effect / denom if denom > 0 else np.nan
            significant_effects[label] = bool(anova_res.loc[raw_name, "PR(>F)"] < 0.05)

    anova_table = anova_res.reset_index().rename(columns={"index": "Effect", "PR(>F)": "p_value"})
    cell_means = subset.groupby([factor_a, factor_b])[numeric_col].mean().unstack()

    return TwoWayANOVAResult(
        dependent_var=numeric_col,
        factor_a=factor_a,
        factor_b=factor_b,
        anova_table=anova_table,
        effect_sizes=effect_sizes,
        significant_effects=significant_effects,
        cell_means=cell_means
    )


def tukey_hsd_posthoc(
    df: pd.DataFrame,
    numeric_col: str,
    group_col: str
) -> TukeyHSDResult:
    """Run Tukey's HSD comparisons for all group pairs."""

    subset = df[[numeric_col, group_col]].dropna()
    if subset[group_col].nunique() < 2:
        raise ValueError("Need at least two groups for Tukey's HSD")

    tukey = pairwise_tukeyhsd(subset[numeric_col], subset[group_col])
    raw_summary = tukey.summary()
    summary_df = pd.DataFrame(raw_summary.data[1:], columns=raw_summary.data[0])
    significant_pairs = [
        (row["group1"], row["group2"])
        for _, row in summary_df.iterrows()
        if row["reject"]
    ]

    return TukeyHSDResult(
        group_col=group_col,
        summary=summary_df,
        significant_pairs=significant_pairs
    )


def chi_square_test(
    df: pd.DataFrame,
    variable1: str,
    variable2: str
) -> ChiSquareResult:
    """Perform chi-square test of independence between two categorical variables."""

    subset = df[[variable1, variable2]].dropna()
    if subset.empty:
        raise ValueError("No overlapping data for chi-square test")

    contingency = pd.crosstab(subset[variable1], subset[variable2])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        raise ValueError("Need at least a 2x2 contingency table")

    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    n = contingency.to_numpy().sum()
    denom = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
    cramers_v = np.sqrt((chi2 / n) / denom) if n > 0 and denom > 0 else np.nan
    expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)

    return ChiSquareResult(
        variable1=variable1,
        variable2=variable2,
        chi_square=chi2,
        p_value=p_val,
        dof=dof,
        significant=bool(p_val < 0.05),
        cramers_v=cramers_v,
        contingency_table=contingency,
        expected_frequencies=expected_df
    )


def levene_test(
    df: pd.DataFrame,
    numeric_col: str,
    group_col: str,
    groups: Optional[List[Any]] = None
) -> LeveneResult:
    """Run Levene's test for equality of variances across groups."""

    subset = df[[numeric_col, group_col]].dropna()
    if groups:
        subset = subset[subset[group_col].isin(groups)]

    grouped = [grp[numeric_col].values for _, grp in subset.groupby(group_col)]
    grouped = [vals for vals in grouped if len(vals) >= 2]

    if len(grouped) < 2:
        raise ValueError("Need at least two groups with 2+ observations for Levene's test")

    stat, p_val = stats.levene(*grouped, center="median")
    total_n = sum(len(vals) for vals in grouped)
    k = len(grouped)
    effect_size = None
    if total_n > k and stat >= 0:
        numerator = stat * (k - 1)
        denominator = numerator + (total_n - k)
        effect_size = numerator / denominator if denominator > 0 else None

    return LeveneResult(
        group_col=group_col,
        numeric_col=numeric_col,
        f_statistic=stat,
        p_value=p_val,
        significant=bool(p_val < 0.05),
        effect_size=effect_size
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


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Identify outliers using configurable Z-score threshold."""

    if threshold <= 0:
        raise ValueError("Threshold must be positive for Z-score detection")

    clean_series = series.dropna()
    if clean_series.empty:
        return clean_series.iloc[0:0]

    std = clean_series.std(ddof=0)
    if np.isnan(std) or np.isclose(std, 0):
        return clean_series.iloc[0:0]

    mean = clean_series.mean()
    z_scores = (clean_series - mean) / std
    abs_z = z_scores.abs()
    mask = abs_z > threshold

    if not mask.any() and len(clean_series) <= 25:
        median = clean_series.median()
        mad = np.median(np.abs(clean_series - median))
        if not np.isnan(mad) and not np.isclose(mad, 0):
            modified_z = 0.6745 * (clean_series - median) / mad
            mask = np.abs(modified_z) > threshold

    return clean_series[mask]
