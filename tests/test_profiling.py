"""Tests for data profiling module."""

import pytest
import pandas as pd
import numpy as np

from core.profiling import (
    detect_column_type,
    profile_column,
    profile_dataset,
    detect_outliers_iqr,
    identify_data_quality_issues
)


class TestDetectColumnType:
    """Tests for column type detection."""
    
    def test_detect_numeric(self):
        series = pd.Series([1, 2, 3, 4, 5])
        assert detect_column_type(series) == "numeric"
    
    def test_detect_categorical(self):
        series = pd.Series(['A', 'B', 'A', 'C', 'B'] * 10)
        assert detect_column_type(series) == "categorical"
    
    def test_detect_boolean(self):
        series = pd.Series([True, False, True, False])
        assert detect_column_type(series) == "boolean"
    
    def test_detect_datetime(self):
        series = pd.Series(pd.date_range('2023-01-01', periods=10))
        assert detect_column_type(series) == "datetime"
    
    def test_detect_text_high_cardinality(self):
        series = pd.Series([f'text_{i}' for i in range(100)])
        assert detect_column_type(series) == "text"


class TestProfileColumn:
    """Tests for column profiling."""
    
    def test_profile_numeric_column(self, sample_numeric_df):
        profile = profile_column(sample_numeric_df['var1'])
        
        assert profile.name == 'var1'
        assert profile.detected_type == 'numeric'
        assert profile.count == 100
        assert profile.null_count == 0
        assert profile.numeric_stats is not None
        assert 'mean' in profile.numeric_stats
        assert 'median' in profile.numeric_stats
        assert 'std' in profile.numeric_stats
    
    def test_profile_categorical_column(self, sample_mixed_df):
        profile = profile_column(sample_mixed_df['categorical'])
        
        assert profile.detected_type == 'categorical'
        assert profile.categorical_stats is not None
        assert 'mode' in profile.categorical_stats
        assert 'top_values' in profile.categorical_stats
    
    def test_profile_with_nulls(self, sample_df_with_nulls):
        profile = profile_column(sample_df_with_nulls['many_nulls'])
        
        assert profile.null_count > 0
        assert profile.null_percentage > 0
        assert len(profile.quality_issues) > 0


class TestDetectOutliers:
    """Tests for outlier detection."""
    
    def test_detect_outliers_iqr(self):
        # Create data with obvious outliers
        data = pd.Series([1, 2, 3, 4, 5, 100, 200])
        outliers = detect_outliers_iqr(data)
        
        assert len(outliers) > 0
        assert 100 in outliers or 200 in outliers
    
    def test_no_outliers_normal_data(self, sample_numeric_df):
        outliers = detect_outliers_iqr(sample_numeric_df['var1'])
        # Normal distribution should have few outliers
        assert len(outliers) < 10


class TestProfileDataset:
    """Tests for dataset profiling."""
    
    def test_profile_complete_dataset(self, sample_mixed_df):
        profile = profile_dataset(sample_mixed_df, "Test Dataset")
        
        assert profile.name == "Test Dataset"
        assert profile.row_count == len(sample_mixed_df)
        assert profile.column_count == len(sample_mixed_df.columns)
        assert len(profile.columns) == len(sample_mixed_df.columns)
        assert profile.completeness_score > 0
    
    def test_profile_dataset_with_nulls(self, sample_df_with_nulls):
        profile = profile_dataset(sample_df_with_nulls)
        
        assert profile.completeness_score < 1.0
        assert len(profile.quality_issues) > 0
    
    def test_column_type_counts(self, sample_mixed_df):
        profile = profile_dataset(sample_mixed_df)
        
        assert 'numeric' in profile.column_types
        assert 'categorical' in profile.column_types
        assert 'boolean' in profile.column_types


class TestDataQualityIssues:
    """Tests for data quality detection."""
    
    def test_identify_empty_columns(self):
        df = pd.DataFrame({
            'good': [1, 2, 3],
            'empty': [np.nan, np.nan, np.nan]
        })
        
        issues = identify_data_quality_issues(df)
        assert len(issues) > 0
        assert any(issue['issue'] == 'empty_column' for issue in issues)
    
    def test_identify_high_cardinality(self):
        df = pd.DataFrame({
            'high_card': [f'val_{i}' for i in range(100)]
        })
        
        issues = identify_data_quality_issues(df)
        assert any(issue['issue'] == 'high_cardinality' for issue in issues)
