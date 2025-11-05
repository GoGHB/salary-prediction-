"""
Utility Module
Contains helper functions for the salary prediction project.
"""

import json
import joblib
import os


def load_config(config_path):
    """
    Load configuration from a JSON file.
    
    Parameters:
    -----------
    config_path : str
        Path to the configuration file
        
    Returns:
    --------
    dict
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in configuration file: {config_path}")


def save_config(config, config_path):
    """
    Save configuration to a JSON file.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    config_path : str
        Path to save the configuration file
    """
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {config_path}")


def save_model(model, filepath):
    """
    Save a trained model to disk.
    
    Parameters:
    -----------
    model : object
        The model to save (can be sklearn model or custom object)
    filepath : str
        Path where the model should be saved
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
    except Exception as e:
        raise Exception(f"Error saving model: {str(e)}")


def load_model(filepath):
    """
    Load a trained model from disk.
    
    Parameters:
    -----------
    filepath : str
        Path to the saved model
        
    Returns:
    --------
    object
        Loaded model
    """
    try:
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")


def calculate_salary_range(predicted_salary, confidence_interval=0.1):
    """
    Calculate a salary range based on predicted value and confidence interval.
    
    Parameters:
    -----------
    predicted_salary : float
        The predicted salary value
    confidence_interval : float
        The confidence interval as a percentage (default: 0.1 for ±10%)
        
    Returns:
    --------
    tuple
        (lower_bound, upper_bound)
    """
    margin = predicted_salary * confidence_interval
    lower_bound = predicted_salary - margin
    upper_bound = predicted_salary + margin
    
    return (lower_bound, upper_bound)


def format_currency(amount, currency='USD'):
    """
    Format a number as currency.
    
    Parameters:
    -----------
    amount : float
        The amount to format
    currency : str
        Currency code (default: 'USD')
        
    Returns:
    --------
    str
        Formatted currency string
    """
    symbols = {
        'USD': '$',
        'EUR': '€',
        'GBP': '£',
        'JPY': '¥'
    }
    
    symbol = symbols.get(currency, currency + ' ')
    return f"{symbol}{amount:,.2f}"


def validate_input_data(data, required_columns):
    """
    Validate that input data contains all required columns.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data to validate
    required_columns : list
        List of required column names
        
    Returns:
    --------
    bool
        True if valid, raises ValueError if invalid
    """
    missing_columns = set(required_columns) - set(data.columns)
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True
