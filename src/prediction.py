"""
Prediction Module

This module handles making predictions using trained models.
"""

import numpy as np
import pandas as pd
import joblib
from typing import Union, List, Dict


class SalaryPredictor:
    """
    A class to handle salary predictions using trained models.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the SalaryPredictor.
        
        Args:
            model_path: Path to a saved model (optional)
        """
        self.model = None
        self.model_path = model_path
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
        """
        try:
            self.model = joblib.load(model_path)
            self.model_path = model_path
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            X: Input features (numpy array or pandas DataFrame)
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first using load_model().")
        
        try:
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            raise Exception(f"Error making predictions: {str(e)}")
    
    def predict_single(self, features: Dict[str, float]) -> float:
        """
        Make a prediction for a single instance.
        
        Args:
            features: Dictionary of feature names and values
            
        Returns:
            Predicted salary
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first using load_model().")
        
        # Convert dictionary to DataFrame for prediction
        X = pd.DataFrame([features])
        prediction = self.predict(X)[0]
        
        return prediction
    
    def predict_batch(self, X: Union[np.ndarray, pd.DataFrame]) -> List[float]:
        """
        Make predictions for multiple instances.
        
        Args:
            X: Input features
            
        Returns:
            List of predictions
        """
        predictions = self.predict(X)
        return predictions.tolist()
    
    def predict_with_confidence(self, X: Union[np.ndarray, pd.DataFrame],
                                n_estimators: int = None) -> Dict[str, np.ndarray]:
        """
        Make predictions with confidence intervals for ensemble models.
        
        Args:
            X: Input features
            n_estimators: Number of estimators to use (for ensemble models)
            
        Returns:
            Dictionary containing predictions, lower bounds, and upper bounds
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first using load_model().")
        
        # Check if model is an ensemble model
        if not hasattr(self.model, 'estimators_'):
            raise ValueError("Confidence intervals are only available for ensemble models.")
        
        # Get predictions from individual estimators
        predictions = []
        for estimator in self.model.estimators_:
            pred = estimator.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate mean and confidence intervals
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        lower_bound = mean_pred - 1.96 * std_pred
        upper_bound = mean_pred + 1.96 * std_pred
        
        return {
            'predictions': mean_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'std': std_pred
        }
    
    def explain_prediction(self, features: Dict[str, float],
                          feature_names: List[str] = None) -> pd.DataFrame:
        """
        Provide basic explanation for a prediction (for tree-based models).
        
        Args:
            features: Dictionary of feature names and values
            feature_names: List of feature names (optional)
            
        Returns:
            DataFrame with feature contributions
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first using load_model().")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Feature importance is only available for tree-based models.")
        
        # Get feature importance
        importances = self.model.feature_importances_
        
        # Create DataFrame
        if feature_names is None:
            feature_names = list(features.keys())
        
        explanation_df = pd.DataFrame({
            'feature': feature_names,
            'value': [features.get(fn, 0) for fn in feature_names],
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Make prediction
        X = pd.DataFrame([features])
        prediction = self.predict(X)[0]
        
        print(f"\nPredicted Salary: ${prediction:,.2f}")
        print("\nFeature Contributions (by importance):")
        print(explanation_df.to_string(index=False))
        
        return explanation_df
    
    def bulk_predict_from_file(self, input_filepath: str,
                              output_filepath: str = None) -> pd.DataFrame:
        """
        Make predictions for data in a CSV file.
        
        Args:
            input_filepath: Path to input CSV file
            output_filepath: Path to save predictions (optional)
            
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first using load_model().")
        
        # Load data
        try:
            df = pd.read_csv(input_filepath)
            print(f"Loaded {len(df)} records from {input_filepath}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
        
        # Make predictions
        predictions = self.predict(df)
        df['predicted_salary'] = predictions
        
        # Save if output path provided
        if output_filepath:
            df.to_csv(output_filepath, index=False)
            print(f"Predictions saved to {output_filepath}")
        
        return df
