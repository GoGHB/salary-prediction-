"""
Data Preprocessing Module

This module handles all data preprocessing tasks including:
- Loading raw data
- Data cleaning
- Feature engineering
- Data transformation
- Train-test splitting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Optional


class DataPreprocessor:
    """
    A class to handle data preprocessing for salary prediction.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the DataPreprocessor.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            DataFrame containing the loaded data
        """
        try:
            df = pd.read_csv(filepath)
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data by handling missing values and outliers.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Handle missing values
        # Numeric columns: fill with median
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # Categorical columns: fill with mode
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        print(f"Data cleaned. Remaining nulls: {df_clean.isnull().sum().sum()}")
        return df_clean
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                    categorical_columns: List[str]) -> pd.DataFrame:
        """
        Encode categorical features using Label Encoding.
        
        Args:
            df: Input DataFrame
            categorical_columns: List of categorical column names
            
        Returns:
            DataFrame with encoded categorical features
        """
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col])
                else:
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
        
        print(f"Encoded {len(categorical_columns)} categorical features")
        return df_encoded
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            X: Input features
            fit: Whether to fit the scaler (True for training data)
            
        Returns:
            Scaled feature array
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def prepare_features(self, df: pd.DataFrame, 
                        target_column: str,
                        categorical_columns: Optional[List[str]] = None) -> Tuple:
        """
        Prepare features for model training.
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            categorical_columns: List of categorical column names
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        df_processed = df.copy()
        
        # Encode categorical features if specified
        if categorical_columns:
            df_processed = self.encode_categorical_features(df_processed, categorical_columns)
        
        # Separate features and target
        y = df_processed[target_column]
        X = df_processed.drop(columns=[target_column])
        
        self.feature_names = X.columns.tolist()
        
        return X, y, self.feature_names
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2) -> Tuple:
        """
        Split data into training and testing sets.
        
        Args:
            X: Features
            y: Target variable
            test_size: Proportion of data to use for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self, filepath: str, 
                           target_column: str,
                           categorical_columns: Optional[List[str]] = None,
                           test_size: float = 0.2,
                           scale: bool = True) -> Tuple:
        """
        Complete preprocessing pipeline.
        
        Args:
            filepath: Path to the data file
            target_column: Name of the target column
            categorical_columns: List of categorical column names
            test_size: Proportion of data to use for testing
            scale: Whether to scale the features
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Load and clean data
        df = self.load_data(filepath)
        df = self.clean_data(df)
        
        # Prepare features
        X, y, _ = self.prepare_features(df, target_column, categorical_columns)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size)
        
        # Scale features if requested
        if scale:
            X_train = self.scale_features(X_train, fit=True)
            X_test = self.scale_features(X_test, fit=False)
        
        return X_train, X_test, y_train, y_test
