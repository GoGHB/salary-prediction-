# Salary Prediction Project

A machine learning project for predicting salaries based on various features such as experience, education, location, and job title.

## Project Overview

This project provides a complete framework for building, training, and deploying salary prediction models. It includes data preprocessing, multiple ML algorithms, model evaluation, and prediction capabilities.

## Features

- **Data Processing**: Automated data cleaning, encoding, and scaling
- **Multiple Models**: Support for Random Forest, Gradient Boosting, Linear Regression, Ridge, and Lasso
- **Easy Configuration**: JSON-based configuration for model parameters
- **Feature Importance**: Analyze which features most impact salary predictions
- **Model Persistence**: Save and load trained models
- **Demo Notebook**: Interactive Jupyter notebook for experimentation

## Project Structure

```
salary-prediction-/
├── src/
│   └── salary_predictor/
│       ├── __init__.py
│       ├── data_processor.py    # Data preprocessing and feature engineering
│       ├── model.py              # ML model implementations
│       └── utils.py              # Utility functions
├── data/
│   ├── raw/                      # Raw data files
│   └── processed/                # Processed data files
├── models/                       # Saved model files
├── notebooks/
│   └── salary_prediction_demo.ipynb  # Demo notebook
├── tests/                        # Unit tests
├── config.json                   # Configuration file
├── train.py                      # Training script
├── predict.py                    # Prediction script
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore file
└── README.md                     # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/GoGHB/salary-prediction-.git
cd salary-prediction-
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Your Data

Place your salary data CSV file in `data/raw/salary_data.csv`. The data should include:
- **Target variable**: salary
- **Numerical features**: years_of_experience, age
- **Categorical features**: job_title, company, location, education_level

### 2. Configure the Model

Edit `config.json` to adjust:
- Data paths and column names
- Model type and parameters
- Feature lists
- Preprocessing options

### 3. Train the Model

```bash
python train.py
```

This will:
- Load and preprocess your data
- Train the model
- Evaluate performance
- Save the trained model to `models/`

### 4. Make Predictions

```bash
python predict.py
```

Or use the model programmatically:

```python
from salary_predictor import load_model
import pandas as pd

# Load trained model and processor
model = load_model('models/salary_predictor.joblib')
processor = load_model('models/salary_predictor_processor.joblib')

# Prepare input data
input_data = pd.DataFrame([{
    'years_of_experience': 5,
    'age': 28,
    'job_title': 'Software Engineer',
    'company': 'IBM',
    'location': 'New York',
    'education_level': 'Bachelor'
}])

# Transform and predict
X = processor.transform_new_data(input_data, ['job_title', 'company', 'location', 'education_level'])
predicted_salary = model.predict(X)[0]
print(f"Predicted Salary: ${predicted_salary:,.2f}")
```

### 5. Explore with Jupyter Notebook

```bash
jupyter notebook notebooks/salary_prediction_demo.ipynb
```

## Configuration

The `config.json` file controls all aspects of the project:

```json
{
    "data": {
        "raw_data_path": "data/raw/salary_data.csv",
        "target_column": "salary",
        "categorical_columns": ["job_title", "company", "location", "education_level"],
        "test_size": 0.2,
        "random_state": 42
    },
    "model": {
        "type": "random_forest",
        "params": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        }
    }
}
```

## Supported Models

- **Random Forest**: Ensemble method with multiple decision trees
- **Gradient Boosting**: Sequential ensemble learning
- **Linear Regression**: Simple linear model
- **Ridge Regression**: Linear model with L2 regularization
- **Lasso Regression**: Linear model with L1 regularization

## API Reference

### DataProcessor

```python
processor = DataProcessor()
processor.load_data(filepath)                    # Load CSV data
processor.clean_data(df)                         # Clean and handle missing values
processor.encode_categorical(df, columns)        # Encode categorical variables
processor.scale_features(X)                      # Scale numerical features
processor.prepare_data(df, target_column)        # Complete preprocessing pipeline
```

### SalaryPredictor

```python
model = SalaryPredictor(model_type='random_forest', **params)
model.train(X_train, y_train)                    # Train the model
model.predict(X)                                 # Make predictions
model.evaluate(X_test, y_test)                   # Evaluate performance
model.get_feature_importance(feature_names)      # Get feature importance
```

## Performance Metrics

The model is evaluated using:
- **RMSE** (Root Mean Squared Error): Average prediction error
- **MAE** (Mean Absolute Error): Average absolute prediction error
- **R² Score**: Proportion of variance explained by the model

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Acknowledgments

- Built for IBM salary prediction analysis
- Uses scikit-learn for machine learning capabilities
- Inspired by real-world salary prediction challenges 
