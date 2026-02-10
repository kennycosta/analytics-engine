"""Test fixtures and configuration for pytest."""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_numeric_df():
    """Create sample DataFrame with numeric columns."""
    np.random.seed(42)
    return pd.DataFrame({
        'var1': np.random.normal(100, 15, 100),
        'var2': np.random.normal(50, 10, 100),
        'var3': np.random.exponential(2, 100)
    })


@pytest.fixture
def sample_mixed_df():
    """Create sample DataFrame with mixed types."""
    np.random.seed(42)
    return pd.DataFrame({
        'numeric': np.random.normal(100, 15, 50),
        'categorical': np.random.choice(['A', 'B', 'C'], 50),
        'boolean': np.random.choice([True, False], 50),
        'text': ['text_' + str(i) for i in range(50)],
        'datetime': pd.date_range('2023-01-01', periods=50)
    })


@pytest.fixture
def sample_df_with_nulls():
    """Create sample DataFrame with null values."""
    np.random.seed(42)
    df = pd.DataFrame({
        'complete': np.random.normal(100, 15, 50),
        'partial_nulls': np.random.normal(50, 10, 50),
        'many_nulls': np.random.normal(75, 5, 50)
    })
    
    # Introduce nulls
    df.loc[0:10, 'partial_nulls'] = np.nan
    df.loc[0:30, 'many_nulls'] = np.nan
    
    return df


@pytest.fixture
def sample_categorical_df():
    """Create sample DataFrame for categorical analysis."""
    np.random.seed(42)
    return pd.DataFrame({
        'value': np.random.normal(100, 20, 100),
        'group': np.random.choice(['Group1', 'Group2', 'Group3'], 100),
        'category': np.random.choice(['Cat_A', 'Cat_B'], 100)
    })
