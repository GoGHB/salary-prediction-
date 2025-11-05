"""
Utility Functions Module

This module contains utility functions for the salary prediction project.
"""

import os
import yaml
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import numpy as np


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        raise Exception(f"Error loading configuration: {str(e)}")


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration file
    """
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Configuration saved to {config_path}")
    except Exception as e:
        raise Exception(f"Error saving configuration: {str(e)}")


def plot_feature_importance(importance_df: pd.DataFrame, 
                           top_n: int = 10,
                           figsize: tuple = (10, 6),
                           save_path: str = None) -> None:
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to plot
        figsize: Figure size
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=figsize)
    
    # Get top N features
    top_features = importance_df.head(top_n)
    
    # Create bar plot
    sns.barplot(x='importance', y='feature', data=top_features, palette='viridis')
    plt.title(f'Top {top_n} Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                               figsize: tuple = (10, 6),
                               save_path: str = None) -> None:
    """
    Plot predicted vs actual values.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        figsize: Figure size
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=figsize)
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Salary')
    plt.ylabel('Predicted Salary')
    plt.title('Predicted vs Actual Salaries')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                   figsize: tuple = (12, 5),
                   save_path: str = None) -> None:
    """
    Plot residuals analysis.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        figsize: Figure size
        save_path: Path to save the plot (optional)
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Residual plot
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Salary')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residual Plot')
    
    # Residual distribution
    axes[1].hist(residuals, bins=30, edgecolor='black')
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def calculate_statistics(data: pd.Series) -> Dict[str, float]:
    """
    Calculate basic statistics for a series.
    
    Args:
        data: Pandas Series
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'mean': data.mean(),
        'median': data.median(),
        'std': data.std(),
        'min': data.min(),
        'max': data.max(),
        'q25': data.quantile(0.25),
        'q75': data.quantile(0.75)
    }
    return stats


def print_data_summary(df: pd.DataFrame) -> None:
    """
    Print a comprehensive summary of the dataset.
    
    Args:
        df: Input DataFrame
    """
    print("=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    print(f"\nDataset Shape: {df.shape}")
    print(f"Number of Rows: {df.shape[0]}")
    print(f"Number of Columns: {df.shape[1]}")
    
    print("\n" + "-" * 80)
    print("Column Information:")
    print("-" * 80)
    print(df.dtypes)
    
    print("\n" + "-" * 80)
    print("Missing Values:")
    print("-" * 80)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0])
    
    print("\n" + "-" * 80)
    print("Numerical Features Summary:")
    print("-" * 80)
    print(df.describe())
    
    print("\n" + "=" * 80)


def save_metrics(metrics: Dict[str, float], filepath: str) -> None:
    """
    Save metrics to a JSON file.
    
    Args:
        metrics: Dictionary of metrics
        filepath: Path to save the metrics
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {filepath}")
    except Exception as e:
        raise Exception(f"Error saving metrics: {str(e)}")


def load_metrics(filepath: str) -> Dict[str, float]:
    """
    Load metrics from a JSON file.
    
    Args:
        filepath: Path to the metrics file
        
    Returns:
        Dictionary of metrics
    """
    try:
        with open(filepath, 'r') as f:
            metrics = json.load(f)
        print(f"Metrics loaded from {filepath}")
        return metrics
    except Exception as e:
        raise Exception(f"Error loading metrics: {str(e)}")


def format_currency(amount: float) -> str:
    """
    Format a number as currency.
    
    Args:
        amount: Amount to format
        
    Returns:
        Formatted currency string
    """
    return f"${amount:,.2f}"


def create_sample_data(n_samples: int = 1000, 
                      output_path: str = None,
                      random_state: int = 42) -> pd.DataFrame:
    """
    Create sample salary data for demonstration.
    
    Args:
        n_samples: Number of samples to generate
        output_path: Path to save the data (optional)
        random_state: Random state for reproducibility
        
    Returns:
        DataFrame with sample data
    """
    # Use modern numpy random generator for better isolation
    rng = np.random.default_rng(random_state)
    
    # Generate features
    data = {
        'years_experience': rng.integers(0, 30, n_samples),
        'education_level': rng.choice(['Bachelor', 'Master', 'PhD'], n_samples),
        'job_title': rng.choice(['Junior', 'Mid-Level', 'Senior', 'Lead'], n_samples),
        'location': rng.choice(['New York', 'San Francisco', 'Austin', 'Seattle', 'Boston'], n_samples),
        'company_size': rng.choice(['Small', 'Medium', 'Large'], n_samples),
        'industry': rng.choice(['Tech', 'Finance', 'Healthcare', 'Retail'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate salary based on features (with some noise)
    base_salary = 50000
    
    # Experience effect
    salary = base_salary + df['years_experience'] * 2000
    
    # Education effect
    education_bonus = df['education_level'].map({
        'Bachelor': 0,
        'Master': 15000,
        'PhD': 30000
    })
    salary += education_bonus
    
    # Job title effect
    title_bonus = df['job_title'].map({
        'Junior': 0,
        'Mid-Level': 20000,
        'Senior': 50000,
        'Lead': 80000
    })
    salary += title_bonus
    
    # Add random noise
    salary += rng.normal(0, 10000, n_samples)
    
    # Ensure no negative salaries
    salary = np.maximum(salary, 30000)
    
    df['salary'] = salary.astype(int)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Sample data saved to {output_path}")
    
    return df
