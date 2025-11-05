"""
Unit tests for the preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import DataPreprocessor
from src.utils import create_sample_data


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return create_sample_data(n_samples=100, random_state=42)


@pytest.fixture
def preprocessor():
    """Create a DataPreprocessor instance."""
    return DataPreprocessor(random_state=42)


def test_data_preprocessor_initialization(preprocessor):
    """Test DataPreprocessor initialization."""
    assert preprocessor.random_state == 42
    assert preprocessor.scaler is not None
    assert isinstance(preprocessor.label_encoders, dict)


def test_clean_data(preprocessor, sample_data):
    """Test data cleaning functionality."""
    df_clean = preprocessor.clean_data(sample_data)
    
    # Check no missing values
    assert df_clean.isnull().sum().sum() == 0
    
    # Check shape is preserved
    assert df_clean.shape == sample_data.shape


def test_encode_categorical_features(preprocessor, sample_data):
    """Test categorical feature encoding."""
    categorical_cols = ['education_level', 'job_title']
    df_encoded = preprocessor.encode_categorical_features(sample_data, categorical_cols)
    
    # Check that categorical columns are encoded
    for col in categorical_cols:
        assert df_encoded[col].dtype in [np.int32, np.int64]


def test_prepare_features(preprocessor, sample_data):
    """Test feature preparation."""
    categorical_cols = ['education_level', 'job_title', 'location', 'company_size', 'industry']
    X, y, feature_names = preprocessor.prepare_features(
        sample_data, 
        target_column='salary',
        categorical_columns=categorical_cols
    )
    
    # Check shapes
    assert len(X) == len(sample_data)
    assert len(y) == len(sample_data)
    assert len(feature_names) == len(X.columns)
    
    # Check target is not in features
    assert 'salary' not in X.columns


def test_split_data(preprocessor, sample_data):
    """Test data splitting."""
    X = sample_data.drop(columns=['salary'])
    y = sample_data['salary']
    
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.2)
    
    # Check sizes
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20


def test_scale_features(preprocessor, sample_data):
    """Test feature scaling."""
    X = sample_data[['years_experience']].values
    
    # Fit and transform
    X_scaled = preprocessor.scale_features(X, fit=True)
    
    # Check that mean is close to 0 and std is close to 1
    assert np.abs(X_scaled.mean()) < 0.1
    assert np.abs(X_scaled.std() - 1.0) < 0.1
