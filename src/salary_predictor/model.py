"""
Model Module
Contains the SalaryPredictor class for training and making predictions.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class SalaryPredictor:
    """
    A class to train and use machine learning models for salary prediction.
    """
    
    AVAILABLE_MODELS = {
        'random_forest': RandomForestRegressor,
        'gradient_boosting': GradientBoostingRegressor,
        'linear_regression': LinearRegression,
        'ridge': Ridge,
        'lasso': Lasso
    }
    
    def __init__(self, model_type='random_forest', **model_params):
        """
        Initialize the SalaryPredictor.
        
        Parameters:
        -----------
        model_type : str
            Type of model to use ('random_forest', 'gradient_boosting', 
            'linear_regression', 'ridge', 'lasso')
        **model_params : dict
            Additional parameters to pass to the model
        """
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model type must be one of {list(self.AVAILABLE_MODELS.keys())}")
        
        self.model_type = model_type
        self.model = self._create_model(model_type, model_params)
        self.is_trained = False
        self.feature_importance = None
        
    def _create_model(self, model_type, params):
        """
        Create a model instance based on the specified type.
        
        Parameters:
        -----------
        model_type : str
            Type of model to create
        params : dict
            Model parameters
            
        Returns:
        --------
        sklearn model
            Initialized model
        """
        ModelClass = self.AVAILABLE_MODELS[model_type]
        
        # Set default parameters for each model type
        default_params = {
            'random_forest': {'n_estimators': 100, 'random_state': 42, 'max_depth': 10},
            'gradient_boosting': {'n_estimators': 100, 'random_state': 42, 'max_depth': 5},
            'linear_regression': {},
            'ridge': {'alpha': 1.0, 'random_state': 42},
            'lasso': {'alpha': 1.0, 'random_state': 42}
        }
        
        # Merge default params with user-provided params
        model_params = default_params.get(model_type, {})
        model_params.update(params)
        
        return ModelClass(**model_params)
    
    def train(self, X_train, y_train):
        """
        Train the model on the provided data.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target values
            
        Returns:
        --------
        self
        """
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Store feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        
        print("Model training completed.")
        return self
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : array-like
            Features to predict on
            
        Returns:
        --------
        numpy.ndarray
            Predicted values
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X_test : array-like
            Test features
        y_test : array-like
            Test target values
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation.")
        
        y_pred = self.predict(X_test)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2_score': r2_score(y_test, y_pred)
        }
        
        print("\nModel Evaluation Metrics:")
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"MAE: {metrics['mae']:.2f}")
        print(f"R² Score: {metrics['r2_score']:.4f}")
        
        return metrics
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance if available.
        
        Parameters:
        -----------
        feature_names : list, optional
            Names of features
            
        Returns:
        --------
        dict or None
            Dictionary mapping feature names to importance scores
        """
        if self.feature_importance is None:
            print("Feature importance not available for this model type.")
            return None
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importance))]
        
        importance_dict = dict(zip(feature_names, self.feature_importance))
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return importance_dict
    
    def get_model(self):
        """
        Get the underlying sklearn model.
        
        Returns:
        --------
        sklearn model
            The trained model
        """
        return self.model
