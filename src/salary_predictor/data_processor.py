"""
Data Processor Module
Handles data loading, cleaning, and preprocessing for salary prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataProcessor:
    """
    A class to handle data preprocessing for salary prediction.
    """
    
    def __init__(self):
        """Initialize the DataProcessor with scalers and encoders."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_data(self, filepath):
        """
        Load data from a CSV file.
        
        Parameters:
        -----------
        filepath : str
            Path to the CSV file
            
        Returns:
        --------
        pandas.DataFrame
            Loaded data
        """
        try:
            data = pd.read_csv(filepath)
            print(f"Data loaded successfully. Shape: {data.shape}")
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def clean_data(self, df):
        """
        Clean the data by handling missing values and duplicates.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
            
        Returns:
        --------
        pandas.DataFrame
            Cleaned dataframe
        """
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        # For numeric columns, fill with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # For categorical columns, fill with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df[col].isnull().any():
                mode_val = df[col].mode()
                fill_value = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                df[col] = df[col].fillna(fill_value)
        
        print(f"Data cleaned. Shape after cleaning: {df.shape}")
        return df
    
    def encode_categorical(self, df, categorical_columns, fit=True):
        """
        Encode categorical variables using Label Encoding.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        categorical_columns : list
            List of categorical column names
        fit : bool
            Whether to fit the encoders (True for training data)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with encoded categorical variables
        """
        df = df.copy()
        
        for col in categorical_columns:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        # Handle unseen labels
                        le = self.label_encoders[col]
                        df[col] = df[col].astype(str).apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
        
        return df
    
    def scale_features(self, X, fit=True):
        """
        Scale numerical features using StandardScaler.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Features to scale
        fit : bool
            Whether to fit the scaler (True for training data)
            
        Returns:
        --------
        numpy.ndarray
            Scaled features
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self.feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def prepare_data(self, df, target_column, categorical_columns=None, test_size=0.2, random_state=42):
        """
        Prepare data for model training.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        target_column : str
            Name of the target column
        categorical_columns : list, optional
            List of categorical column names
        test_size : float
            Proportion of data to use for testing
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        # Clean the data
        df = self.clean_data(df)
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Encode categorical variables
        if categorical_columns:
            X = self.encode_categorical(X, categorical_columns, fit=True)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train = self.scale_features(X_train, fit=True)
        X_test = self.scale_features(X_test, fit=False)
        
        print(f"Training set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def transform_new_data(self, df, categorical_columns=None):
        """
        Transform new data using fitted encoders and scalers.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            New data to transform
        categorical_columns : list, optional
            List of categorical column names
            
        Returns:
        --------
        numpy.ndarray
            Transformed data
        """
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Encode categorical variables
        if categorical_columns:
            df = self.encode_categorical(df, categorical_columns, fit=False)
        
        # Ensure columns are in the same order as training
        if self.feature_names is not None:
            df = df[self.feature_names]
        
        # Scale features
        X_scaled = self.scale_features(df, fit=False)
        
        return X_scaled
