# Models Directory

This directory stores trained machine learning models.

## File Naming Convention

Models are saved with descriptive names including:
- Model type (e.g., random_forest, gradient_boosting)
- Date/timestamp
- Performance metrics (optional)

Example: `salary_prediction_model.pkl`

## Usage

Models can be loaded using joblib:
```python
import joblib
model = joblib.load('models/salary_prediction_model.pkl')
```
