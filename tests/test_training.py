"""
Unit tests for the training module.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training import ModelTrainer
from src.utils import create_sample_data
from src.preprocessing import DataPreprocessor


@pytest.fixture
def sample_training_data():
    """Create sample data for training tests."""
    df = create_sample_data(n_samples=100, random_state=42)
    
    # Process data directly without using preprocess_pipeline
    preprocessor = DataPreprocessor(random_state=42)
    categorical_cols = ['education_level', 'job_title', 'location', 'company_size', 'industry']
    df_encoded = preprocessor.encode_categorical_features(df, categorical_cols)
    X, y, _ = preprocessor.prepare_features(df_encoded, 'salary')
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.2)
    X_train = preprocessor.scale_features(X_train, fit=True)
    X_test = preprocessor.scale_features(X_test, fit=False)
    
    return X_train, X_test, y_train, y_test


@pytest.fixture
def trainer():
    """Create a ModelTrainer instance."""
    return ModelTrainer(random_state=42)


def test_model_trainer_initialization(trainer):
    """Test ModelTrainer initialization."""
    assert trainer.random_state == 42
    assert trainer.model is None


def test_get_model(trainer):
    """Test getting different model types."""
    # Test linear regression
    model = trainer.get_model('linear')
    assert model is not None
    
    # Test random forest
    model = trainer.get_model('random_forest', n_estimators=50)
    assert model is not None
    
    # Test invalid model type
    with pytest.raises(ValueError):
        trainer.get_model('invalid_model')


def test_train_model(trainer, sample_training_data):
    """Test model training."""
    X_train, X_test, y_train, y_test = sample_training_data
    
    model = trainer.train_model(X_train, y_train, model_type='linear')
    
    assert model is not None
    assert trainer.model is not None
    assert trainer.model_name == 'linear'


def test_evaluate_model(trainer, sample_training_data):
    """Test model evaluation."""
    X_train, X_test, y_train, y_test = sample_training_data
    
    # Train a model first
    trainer.train_model(X_train, y_train, model_type='linear')
    
    # Evaluate
    metrics = trainer.evaluate_model(X_test, y_test)
    
    # Check that all metrics are present
    assert 'mse' in metrics
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics
    
    # Check that metrics are reasonable
    assert metrics['rmse'] > 0
    assert 0 <= metrics['r2'] <= 1


def test_evaluate_without_training(trainer, sample_training_data):
    """Test that evaluation fails without training."""
    X_train, X_test, y_train, y_test = sample_training_data
    
    with pytest.raises(ValueError):
        trainer.evaluate_model(X_test, y_test)
