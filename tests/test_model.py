"""
Unit tests for the SalaryPredictor class.
"""

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from salary_predictor import SalaryPredictor


class TestSalaryPredictor(unittest.TestCase):
    """Test cases for SalaryPredictor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample training data
        np.random.seed(42)
        self.X_train = np.random.randn(100, 5)
        self.y_train = np.random.randn(100) * 10000 + 50000
        self.X_test = np.random.randn(20, 5)
        self.y_test = np.random.randn(20) * 10000 + 50000
    
    def test_initialization(self):
        """Test SalaryPredictor initialization."""
        model = SalaryPredictor(model_type='random_forest')
        self.assertEqual(model.model_type, 'random_forest')
        self.assertFalse(model.is_trained)
        self.assertIsNone(model.feature_importance)
    
    def test_invalid_model_type(self):
        """Test that invalid model type raises error."""
        with self.assertRaises(ValueError):
            SalaryPredictor(model_type='invalid_model')
    
    def test_train(self):
        """Test model training."""
        model = SalaryPredictor(model_type='random_forest')
        model.train(self.X_train, self.y_train)
        
        self.assertTrue(model.is_trained)
        self.assertIsNotNone(model.feature_importance)
    
    def test_predict_before_training(self):
        """Test that prediction before training raises error."""
        model = SalaryPredictor(model_type='random_forest')
        
        with self.assertRaises(ValueError):
            model.predict(self.X_test)
    
    def test_predict_after_training(self):
        """Test prediction after training."""
        model = SalaryPredictor(model_type='random_forest')
        model.train(self.X_train, self.y_train)
        
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(all(isinstance(p, (int, float, np.number)) for p in predictions))
    
    def test_evaluate(self):
        """Test model evaluation."""
        model = SalaryPredictor(model_type='random_forest')
        model.train(self.X_train, self.y_train)
        
        metrics = model.evaluate(self.X_test, self.y_test)
        
        # Check that all metrics are present
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2_score', metrics)
        
        # Check that metrics are numerical
        self.assertTrue(all(isinstance(v, (int, float)) for v in metrics.values()))
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        model = SalaryPredictor(model_type='random_forest')
        model.train(self.X_train, self.y_train)
        
        feature_names = [f'feature_{i}' for i in range(5)]
        importance = model.get_feature_importance(feature_names)
        
        self.assertIsNotNone(importance)
        self.assertEqual(len(importance), 5)
    
    def test_linear_regression_no_feature_importance(self):
        """Test that linear regression doesn't have feature importance."""
        model = SalaryPredictor(model_type='linear_regression')
        model.train(self.X_train, self.y_train)
        
        importance = model.get_feature_importance()
        self.assertIsNone(importance)
    
    def test_different_model_types(self):
        """Test that different model types can be trained."""
        model_types = ['random_forest', 'gradient_boosting', 'linear_regression', 'ridge', 'lasso']
        
        for model_type in model_types:
            model = SalaryPredictor(model_type=model_type)
            model.train(self.X_train, self.y_train)
            predictions = model.predict(self.X_test)
            
            self.assertTrue(model.is_trained)
            self.assertEqual(len(predictions), len(self.X_test))


if __name__ == '__main__':
    unittest.main()
