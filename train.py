"""
Training Script
Script to train the salary prediction model.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from salary_predictor import DataProcessor, SalaryPredictor, load_config, save_model


def main():
    """Main training function."""
    # Load configuration
    print("Loading configuration...")
    config = load_config('config.json')
    
    # Initialize data processor
    print("\nInitializing data processor...")
    processor = DataProcessor()
    
    # Load and prepare data
    print("\nLoading data...")
    data_path = config['data']['raw_data_path']
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please place your salary data CSV file in the data/raw/ directory.")
        print("\nExpected columns:")
        print(f"  - Target: {config['data']['target_column']}")
        print(f"  - Numerical features: {config['features']['numerical']}")
        print(f"  - Categorical features: {config['features']['categorical']}")
        return
    
    try:
        df = processor.load_data(data_path)
        
        # Prepare data
        print("\nPreparing data for training...")
        X_train, X_test, y_train, y_test = processor.prepare_data(
            df,
            target_column=config['data']['target_column'],
            categorical_columns=config['data']['categorical_columns'],
            test_size=config['data']['test_size'],
            random_state=config['data']['random_state']
        )
        
        # Initialize and train model
        print("\nInitializing model...")
        model = SalaryPredictor(
            model_type=config['model']['type'],
            **config['model']['params']
        )
        
        model.train(X_train, y_train)
        
        # Evaluate model
        print("\nEvaluating model...")
        metrics = model.evaluate(X_test, y_test)
        
        # Show feature importance
        if processor.feature_names:
            print("\nFeature Importance:")
            importance = model.get_feature_importance(processor.feature_names)
            if importance:
                for feature, score in list(importance.items())[:10]:
                    print(f"  {feature}: {score:.4f}")
        
        # Save model
        print("\nSaving model...")
        save_model(model, config['model']['save_path'])
        
        # Save processor
        processor_path = config['model']['save_path'].replace('.joblib', '_processor.joblib')
        save_model(processor, processor_path)
        
        print("\n" + "="*50)
        print("Training completed successfully!")
        print("="*50)
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
