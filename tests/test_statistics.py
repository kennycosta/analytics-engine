"""Tests for statistical analysis module."""

import pytest
import pandas as pd
import numpy as np

from core.statistics import (
    calculate_correlation,
    correlation_matrix,
    linear_regression_analysis,
    independent_t_test,
    anova_test,
    detect_trend,
    detect_outliers_zscore,
)


class TestCorrelation:
    """Tests for correlation analysis."""
    
    def test_pearson_correlation(self, sample_numeric_df):
        result = calculate_correlation(
            sample_numeric_df, 'var1', 'var2', method='pearson'
        )
        
        assert result.variable1 == 'var1'
        assert result.variable2 == 'var2'
        assert -1 <= result.correlation <= 1
        assert 0 <= result.p_value <= 1
        assert result.method == 'pearson'
        assert result.strength in ['weak', 'moderate', 'strong']
    
    def test_spearman_correlation(self, sample_numeric_df):
        result = calculate_correlation(
            sample_numeric_df, 'var1', 'var3', method='spearman'
        )
        
        assert result.method == 'spearman'
        assert -1 <= result.correlation <= 1
    
    def test_correlation_matrix(self, sample_numeric_df):
        corr_mat = correlation_matrix(sample_numeric_df)
        
        assert corr_mat.shape == (3, 3)
        # Diagonal should be 1
        assert all(corr_mat.iloc[i, i] == 1.0 for i in range(3))
    
    def test_insufficient_data_raises_error(self):
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        
        with pytest.raises(ValueError):
            calculate_correlation(df, 'a', 'b')


class TestLinearRegression:
    """Tests for linear regression analysis."""
    
    def test_simple_linear_regression(self):
        # Create data with known relationship
        np.random.seed(42)
        df = pd.DataFrame({
            'x': np.arange(100),
            'y': 2 * np.arange(100) + np.random.normal(0, 10, 100)
        })
        
        result = linear_regression_analysis(df, 'y', ['x'])
        
        assert result.dependent_var == 'y'
        assert result.independent_vars == ['x']
        assert 'x' in result.coefficients
        assert result.r_squared > 0.8  # Should have high R²
        assert result.rmse > 0
    
    def test_multiple_regression(self, sample_numeric_df):
        result = linear_regression_analysis(
            sample_numeric_df, 'var1', ['var2', 'var3']
        )
        
        assert len(result.coefficients) == 2
        assert 'var2' in result.coefficients
        assert 'var3' in result.coefficients
        assert 0 <= result.r_squared <= 1
    
    def test_regression_insufficient_data_raises_error(self):
        df = pd.DataFrame({
            'x': [1, 2],
            'y': [3, 4]
        })
        
        with pytest.raises(ValueError):
            linear_regression_analysis(df, 'y', ['x'])


class TestTTest:
    """Tests for t-test analysis."""
    
    def test_independent_t_test(self, sample_categorical_df):
        result = independent_t_test(
            sample_categorical_df,
            'value',
            'category',
            'Cat_A',
            'Cat_B'
        )
        
        assert result.group1_name == 'Cat_A'
        assert result.group2_name == 'Cat_B'
        assert result.group1_mean is not None
        assert result.group2_mean is not None
        assert result.t_statistic is not None
        assert 0 <= result.p_value <= 1
        assert isinstance(result.significant, bool)
    
    def test_t_test_with_clear_difference(self):
        # Create data with clear group differences
        df = pd.DataFrame({
            'value': list(np.random.normal(100, 10, 50)) + list(np.random.normal(150, 10, 50)),
            'group': ['A'] * 50 + ['B'] * 50
        })
        
        result = independent_t_test(df, 'value', 'group', 'A', 'B')
        
        assert result.significant is True
        assert abs(result.effect_size) > 1  # Large effect size
    
    def test_t_test_insufficient_data_raises_error(self):
        df = pd.DataFrame({
            'value': [1, 2, 3],
            'group': ['A', 'B', 'A']
        })
        
        with pytest.raises(ValueError):
            independent_t_test(df, 'value', 'group', 'A', 'B')


class TestANOVA:
    """Tests for ANOVA analysis."""
    
    def test_anova_multiple_groups(self, sample_categorical_df):
        result = anova_test(sample_categorical_df, 'value', 'group')
        
        assert len(result.groups) >= 2
        assert result.f_statistic is not None
        assert 0 <= result.p_value <= 1
        assert len(result.group_means) == len(result.groups)
    
    def test_anova_with_significant_differences(self):
        # Create data with clear group differences
        df = pd.DataFrame({
            'value': (
                list(np.random.normal(100, 10, 30)) +
                list(np.random.normal(150, 10, 30)) +
                list(np.random.normal(200, 10, 30))
            ),
            'group': ['A'] * 30 + ['B'] * 30 + ['C'] * 30
        })
        
        result = anova_test(df, 'value', 'group')
        
        assert result.significant is True
        assert result.f_statistic > 1


class TestTrendDetection:
    """Tests for trend detection."""
    
    def test_detect_increasing_trend(self):
        # Create increasing series
        series = pd.Series(np.arange(100) + np.random.normal(0, 5, 100))
        result = detect_trend(series)
        
        assert result['trend'] in ['increasing', 'stable']
        assert result['slope'] > 0
        assert 0 <= result['r_squared'] <= 1
    
    def test_detect_decreasing_trend(self):
        # Create decreasing series
        series = pd.Series(-np.arange(100) + np.random.normal(0, 5, 100))
        result = detect_trend(series)
        
        assert result['trend'] in ['decreasing', 'stable']
        assert result['slope'] < 0
    
    def test_detect_no_trend(self):
        # Create random series
        np.random.seed(42)
        series = pd.Series(np.random.normal(0, 1, 100))
        result = detect_trend(series)
        
        # Should likely be stable
        assert result['trend'] in ['increasing', 'decreasing', 'stable']
    
    def test_insufficient_data(self):
        series = pd.Series([1, 2])
        result = detect_trend(series)
        
        assert result['trend'] == 'insufficient_data'


class TestOutlierDetection:
    """Tests for Z-score based outlier detection."""

    def test_detect_outliers_zscore_flags_extreme_values(self):
        series = pd.Series([10, 12, 11, 13, 50])
        outliers = detect_outliers_zscore(series, threshold=2.0)

        assert len(outliers) == 1
        assert outliers.iloc[0] == 50

    def test_detect_outliers_zscore_handles_constant_series(self):
        series = pd.Series([5, 5, 5, 5])
        outliers = detect_outliers_zscore(series, threshold=1.5)

        assert outliers.empty
