"""
Model Training Module

This module handles model training, evaluation, and hyperparameter tuning.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from typing import Dict, Any, Tuple, Optional
import os


class ModelTrainer:
    """
    A class to handle model training and evaluation for salary prediction.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the ModelTrainer.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.model_name = None
        self.training_metrics = {}
        
    def get_model(self, model_type: str, **kwargs) -> Any:
        """
        Get a model instance based on the model type.
        
        Args:
            model_type: Type of model ('linear', 'ridge', 'lasso', 'random_forest', 'gradient_boosting')
            **kwargs: Additional parameters for the model
            
        Returns:
            Model instance
        """
        if model_type == 'linear':
            return LinearRegression()
        elif model_type == 'ridge':
            return Ridge(random_state=self.random_state, **kwargs)
        elif model_type == 'lasso':
            return Lasso(random_state=self.random_state, **kwargs)
        elif model_type == 'random_forest':
            return RandomForestRegressor(random_state=self.random_state, **kwargs)
        elif model_type == 'gradient_boosting':
            return GradientBoostingRegressor(random_state=self.random_state, **kwargs)
        else:
            raise ValueError(f"Model type '{model_type}' not supported. Choose from: ['linear', 'ridge', 'lasso', 'random_forest', 'gradient_boosting']")
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   model_type: str = 'random_forest', **kwargs) -> Any:
        """
        Train a model on the training data.
        
        Args:
            X_train: Training features
            y_train: Training target
            model_type: Type of model to train
            **kwargs: Additional parameters for the model
            
        Returns:
            Trained model
        """
        self.model = self.get_model(model_type, **kwargs)
        self.model_name = model_type
        
        print(f"Training {model_type} model...")
        self.model.fit(X_train, y_train)
        print("Training completed!")
        
        return self.model
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("No model has been trained yet. Call train_model first.")
        
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        print("\nModel Evaluation Metrics:")
        print(f"  MSE (Mean Squared Error): {metrics['mse']:.2f}")
        print(f"  RMSE (Root Mean Squared Error): {metrics['rmse']:.2f}")
        print(f"  MAE (Mean Absolute Error): {metrics['mae']:.2f}")
        print(f"  R² Score: {metrics['r2']:.4f}")
        
        self.training_metrics = metrics
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation on the model.
        
        Args:
            X: Features
            y: Target
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary containing cross-validation results
        """
        if self.model is None:
            raise ValueError("No model has been trained yet. Call train_model first.")
        
        cv_scores = cross_val_score(self.model, X, y, cv=cv, 
                                   scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)
        
        results = {
            'cv_scores': cv_scores,
            'cv_rmse': cv_rmse,
            'mean_rmse': cv_rmse.mean(),
            'std_rmse': cv_rmse.std()
        }
        
        print(f"\nCross-Validation Results ({cv} folds):")
        print(f"  Mean RMSE: {results['mean_rmse']:.2f} (+/- {results['std_rmse']:.2f})")
        
        return results
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                             model_type: str, param_grid: Dict[str, Any],
                             cv: int = 5) -> Tuple[Any, Dict[str, Any]]:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training target
            model_type: Type of model to tune
            param_grid: Dictionary of parameters to search
            cv: Number of cross-validation folds
            
        Returns:
            Tuple of (best_model, best_params)
        """
        base_model = self.get_model(model_type)
        
        print(f"Starting hyperparameter tuning for {model_type}...")
        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.model_name = model_type
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score (RMSE): {np.sqrt(-grid_search.best_score_):.2f}")
        
        return self.model, grid_search.best_params_
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Get feature importance for tree-based models.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("No model has been trained yet.")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("This model does not support feature importance.")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(importance_df.to_string(index=False))
        
        return importance_df
    
    def save_model(self, filepath: str, create_dir: bool = True) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path where to save the model
            create_dir: Whether to create the directory if it doesn't exist
        """
        if self.model is None:
            raise ValueError("No model has been trained yet.")
        
        if create_dir:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> Any:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model
        """
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return self.model
    
    def compare_models(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      model_types: list = None) -> pd.DataFrame:
        """
        Compare multiple models and return their performance metrics.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            model_types: List of model types to compare
            
        Returns:
            DataFrame with comparison results
        """
        if model_types is None:
            model_types = ['linear', 'ridge', 'lasso', 'random_forest', 'gradient_boosting']
        
        results = []
        
        for model_type in model_types:
            print(f"\nEvaluating {model_type}...")
            self.train_model(X_train, y_train, model_type)
            metrics = self.evaluate_model(X_test, y_test)
            
            results.append({
                'model': model_type,
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'r2': metrics['r2']
            })
        
        comparison_df = pd.DataFrame(results).sort_values('rmse')
        
        print("\n" + "="*60)
        print("Model Comparison Results:")
        print("="*60)
        print(comparison_df.to_string(index=False))
        
        return comparison_df
