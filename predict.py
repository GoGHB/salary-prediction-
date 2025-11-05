"""
Prediction Script
Script to make salary predictions using a trained model.
"""

import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from salary_predictor import load_model, load_config, format_currency, calculate_salary_range


def predict_salary(features_dict, model_path, processor_path):
    """
    Make a salary prediction for given features.
    
    Parameters:
    -----------
    features_dict : dict
        Dictionary containing feature values
    model_path : str
        Path to the saved model
    processor_path : str
        Path to the saved processor
        
    Returns:
    --------
    float
        Predicted salary
    """
    # Load model and processor
    model = load_model(model_path)
    processor = load_model(processor_path)
    
    # Convert features to DataFrame
    df = pd.DataFrame([features_dict])
    
    # Process the data
    config = load_config('config.json')
    X = processor.transform_new_data(df, config['data']['categorical_columns'])
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    return prediction


def main():
    """Main prediction function."""
    print("Salary Prediction System")
    print("="*50)
    
    # Load configuration
    config = load_config('config.json')
    
    model_path = config['model']['save_path']
    processor_path = model_path.replace('.joblib', '_processor.joblib')
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first by running: python train.py")
        return
    
    # Example prediction
    print("\nExample Prediction:")
    print("-"*50)
    
    # Create sample input
    sample_features = {
        'job_title': 'Software Engineer',
        'company': 'IBM',
        'location': 'New York',
        'education_level': 'Bachelor',
        'years_of_experience': 5,
        'age': 28
    }
    
    print("\nInput Features:")
    for key, value in sample_features.items():
        print(f"  {key}: {value}")
    
    try:
        # Make prediction
        predicted_salary = predict_salary(sample_features, model_path, processor_path)
        
        # Calculate salary range
        lower, upper = calculate_salary_range(predicted_salary)
        
        print("\nPrediction Results:")
        print(f"  Predicted Salary: {format_currency(predicted_salary)}")
        print(f"  Estimated Range: {format_currency(lower)} - {format_currency(upper)}")
        
    except Exception as e:
        print(f"\nError during prediction: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
