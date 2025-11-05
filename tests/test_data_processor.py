"""
Unit tests for the DataProcessor class.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from salary_predictor import DataProcessor


class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = DataProcessor()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'years_of_experience': [1, 2, 3, 4, 5],
            'age': [22, 25, 28, 30, 35],
            'job_title': ['Engineer', 'Engineer', 'Manager', 'Manager', 'Director'],
            'salary': [50000, 60000, 70000, 80000, 100000]
        })
    
    def test_initialization(self):
        """Test DataProcessor initialization."""
        self.assertIsNotNone(self.processor.scaler)
        self.assertEqual(len(self.processor.label_encoders), 0)
        self.assertIsNone(self.processor.feature_names)
    
    def test_clean_data(self):
        """Test data cleaning functionality."""
        # Create data with missing values
        dirty_data = self.sample_data.copy()
        dirty_data.loc[0, 'age'] = np.nan
        dirty_data.loc[1, 'job_title'] = np.nan
        
        cleaned = self.processor.clean_data(dirty_data)
        
        # Check that no missing values remain
        self.assertEqual(cleaned.isnull().sum().sum(), 0)
        self.assertEqual(len(cleaned), len(dirty_data))
    
    def test_encode_categorical(self):
        """Test categorical encoding."""
        df = self.sample_data.copy()
        encoded = self.processor.encode_categorical(df, ['job_title'], fit=True)
        
        # Check that job_title was encoded
        self.assertTrue(encoded['job_title'].dtype in [np.int32, np.int64])
        self.assertEqual(len(self.processor.label_encoders), 1)
    
    def test_scale_features(self):
        """Test feature scaling."""
        X = self.sample_data[['years_of_experience', 'age']]
        X_scaled = self.processor.scale_features(X, fit=True)
        
        # Check that scaling was applied
        self.assertIsInstance(X_scaled, np.ndarray)
        self.assertEqual(X_scaled.shape, X.shape)
        
        # Check that mean is close to 0 and std is close to 1
        self.assertTrue(np.abs(X_scaled.mean()) < 1e-10)
        self.assertTrue(np.abs(X_scaled.std() - 1.0) < 0.5)
    
    def test_prepare_data(self):
        """Test complete data preparation pipeline."""
        X_train, X_test, y_train, y_test = self.processor.prepare_data(
            self.sample_data,
            target_column='salary',
            categorical_columns=['job_title'],
            test_size=0.2,
            random_state=42
        )
        
        # Check shapes
        self.assertEqual(X_train.shape[0] + X_test.shape[0], len(self.sample_data))
        self.assertEqual(len(y_train) + len(y_test), len(self.sample_data))
        
        # Check that features are scaled
        self.assertIsInstance(X_train, np.ndarray)
        self.assertIsInstance(X_test, np.ndarray)


if __name__ == '__main__':
    unittest.main()
