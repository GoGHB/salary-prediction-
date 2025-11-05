"""
Salary Predictor Package
A machine learning package for predicting salaries based on various features.
"""

__version__ = "0.1.0"
__author__ = "IBM Salary Prediction Team"

from .data_processor import DataProcessor
from .model import SalaryPredictor
from .utils import load_config, save_model, load_model

__all__ = [
    'DataProcessor',
    'SalaryPredictor',
    'load_config',
    'save_model',
    'load_model'
]
