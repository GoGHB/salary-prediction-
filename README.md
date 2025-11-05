# Salary Prediction Project

A machine learning project for predicting salaries based on various professional and demographic features, developed for IBM.

## 📋 Project Overview

This project implements a comprehensive salary prediction system using machine learning techniques. It includes data preprocessing, multiple model training options, prediction capabilities, and visualization tools.

## 🚀 Features

- **Data Preprocessing**: Automated data cleaning, feature engineering, and transformation
- **Multiple Models**: Support for Linear Regression, Ridge, Lasso, Random Forest, and Gradient Boosting
- **Model Training**: Easy-to-use training pipeline with cross-validation
- **Hyperparameter Tuning**: Built-in GridSearchCV for optimal model parameters
- **Predictions**: Single and batch prediction capabilities
- **Visualization**: Feature importance, prediction analysis, and residual plots
- **Configuration**: YAML-based configuration for easy customization

## 📁 Project Structure

```
salary-prediction-/
├── data/
│   ├── raw/              # Raw data files
│   └── processed/        # Processed data files
├── models/               # Saved trained models
├── notebooks/            # Jupyter notebooks for exploration and demos
├── src/                  # Source code
│   ├── __init__.py
│   ├── preprocessing.py  # Data preprocessing module
│   ├── training.py       # Model training module
│   ├── prediction.py     # Prediction module
│   └── utils.py          # Utility functions
├── tests/                # Unit tests
├── config.yaml           # Configuration file
├── requirements.txt      # Project dependencies
├── .gitignore           # Git ignore rules
└── README.md            # Project documentation
```

## 🔧 Installation

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

## 💻 Quick Start

### 1. Generate Sample Data (Optional)

```python
from src.utils import create_sample_data

# Generate 1000 sample records
df = create_sample_data(n_samples=1000, output_path='data/raw/salary_data.csv')
```

### 2. Preprocess Data

```python
from src.preprocessing import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor(random_state=42)

# Run preprocessing pipeline
X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
    filepath='data/raw/salary_data.csv',
    target_column='salary',
    categorical_columns=['education_level', 'job_title', 'location', 'company_size', 'industry'],
    test_size=0.2,
    scale=True
)
```

### 3. Train Model

```python
from src.training import ModelTrainer

# Initialize trainer
trainer = ModelTrainer(random_state=42)

# Train a Random Forest model
model = trainer.train_model(X_train, y_train, model_type='random_forest', n_estimators=100)

# Evaluate the model
metrics = trainer.evaluate_model(X_test, y_test)

# Save the model
trainer.save_model('models/salary_prediction_model.pkl')
```

### 4. Make Predictions

```python
from src.prediction import SalaryPredictor

# Load the trained model
predictor = SalaryPredictor('models/salary_prediction_model.pkl')

# Make a single prediction
features = {
    'years_experience': 5,
    'education_level': 1,  # Encoded: 0=Bachelor, 1=Master, 2=PhD
    'job_title': 2,        # Encoded: 0=Junior, 1=Mid-Level, 2=Senior, 3=Lead
    'location': 1,
    'company_size': 1,
    'industry': 0
}

predicted_salary = predictor.predict_single(features)
print(f"Predicted Salary: ${predicted_salary:,.2f}")
```

## 📊 Usage Examples

### Compare Multiple Models

```python
from src.training import ModelTrainer

trainer = ModelTrainer()
comparison = trainer.compare_models(
    X_train, y_train, X_test, y_test,
    model_types=['linear', 'ridge', 'random_forest', 'gradient_boosting']
)
print(comparison)
```

### Visualize Feature Importance

```python
from src.utils import plot_feature_importance

# Get feature importance
importance_df = trainer.get_feature_importance(feature_names)

# Plot
plot_feature_importance(importance_df, top_n=10, save_path='plots/feature_importance.png')
```

### Batch Predictions from File

```python
from src.prediction import SalaryPredictor

predictor = SalaryPredictor('models/salary_prediction_model.pkl')
results = predictor.bulk_predict_from_file(
    input_filepath='data/raw/new_data.csv',
    output_filepath='data/processed/predictions.csv'
)
```

## ⚙️ Configuration

The project uses a `config.yaml` file for configuration. You can modify settings such as:

- Data paths and parameters
- Model type and hyperparameters
- Training settings
- Feature engineering options
- Evaluation metrics

Example configuration:

```yaml
model:
  type: "random_forest"
  random_forest:
    n_estimators: 100
    max_depth: 20
    min_samples_split: 5

training:
  scale_features: true
  cross_validation_folds: 5
```

## 🧪 Testing

Run tests using pytest:

```bash
pytest tests/
```

## 📈 Model Performance

The project supports multiple evaluation metrics:

- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R² Score**

Example output:
```
Model Evaluation Metrics:
  MSE (Mean Squared Error): 45678912.00
  RMSE (Root Mean Squared Error): 6758.36
  MAE (Mean Absolute Error): 5234.12
  R² Score: 0.8756
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is part of IBM's data science initiatives.

## 👥 Authors

IBM Data Science Team

## 🙏 Acknowledgments

- scikit-learn for machine learning algorithms
- pandas for data manipulation
- matplotlib and seaborn for visualizations

## 📞 Support

For questions or issues, please open an issue on GitHub or contact the development team.

---

**Note**: This is a demonstration project. For production use, please ensure proper data validation, error handling, and security measures are in place. 
