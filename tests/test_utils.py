"""
Unit tests for utility functions.
"""

import unittest
import json
import os
import tempfile
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from salary_predictor import utils


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def test_save_and_load_config(self):
        """Test config saving and loading."""
        config = {
            'model': {'type': 'random_forest'},
            'data': {'test_size': 0.2}
        }
        
        config_path = os.path.join(self.temp_dir, 'test_config.json')
        utils.save_config(config, config_path)
        
        loaded_config = utils.load_config(config_path)
        
        self.assertEqual(config, loaded_config)
    
    def test_load_config_not_found(self):
        """Test loading non-existent config file."""
        with self.assertRaises(FileNotFoundError):
            utils.load_config('nonexistent_config.json')
    
    def test_calculate_salary_range(self):
        """Test salary range calculation."""
        predicted_salary = 100000
        lower, upper = utils.calculate_salary_range(predicted_salary, confidence_interval=0.1)
        
        self.assertEqual(lower, 90000)
        self.assertEqual(upper, 110000)
    
    def test_format_currency_usd(self):
        """Test USD currency formatting."""
        formatted = utils.format_currency(50000, 'USD')
        self.assertEqual(formatted, '$50,000.00')
    
    def test_format_currency_eur(self):
        """Test EUR currency formatting."""
        formatted = utils.format_currency(50000, 'EUR')
        self.assertEqual(formatted, '€50,000.00')
    
    def test_validate_input_data_valid(self):
        """Test validation with valid data."""
        import pandas as pd
        
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'feature3': [7, 8, 9]
        })
        
        required_columns = ['feature1', 'feature2', 'feature3']
        
        # Should not raise an error
        result = utils.validate_input_data(df, required_columns)
        self.assertTrue(result)
    
    def test_validate_input_data_missing_columns(self):
        """Test validation with missing columns."""
        import pandas as pd
        
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        required_columns = ['feature1', 'feature2', 'feature3']
        
        with self.assertRaises(ValueError):
            utils.validate_input_data(df, required_columns)


if __name__ == '__main__':
    unittest.main()
