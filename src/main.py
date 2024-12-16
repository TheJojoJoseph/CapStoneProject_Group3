import os
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict

from phase1_data_exploration import DataExplorer
from phase2_data_processing import DataProcessor
from phase3_feature_engineering import FeatureEngineer
from phase4_model_training import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_directory_structure():
    """Create necessary directories for the project."""
    directories = [
        'data/raw',
        'data/processed',
        'data/interim',
        'models',
        'reports',
        'reports/figures'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def run_phase1(data_paths: Dict[str, str]):
    """Run Phase 1: Data Exploration"""
    logger.info("Starting Phase 1: Data Exploration")
    
    explorer = DataExplorer(data_paths)
    explorer.load_data()
    
    # Analyze data
    explorer.analyze_missing_values()
    explorer.generate_summary_statistics()
    
    # Generate visualizations
    explorer.plot_missing_values('reports/figures')
    explorer.analyze_numerical_distributions('reports/figures')
    explorer.generate_correlation_matrix('reports/figures')
    
    # Save analysis report
    explorer.save_analysis_report('reports/data_exploration_report.txt')
    
    logger.info("Phase 1 completed successfully")
    return explorer.datasets

def run_phase2(data: Dict[str, pd.DataFrame]):
    """Run Phase 2: Data Processing"""
    logger.info("Starting Phase 2: Data Processing")
    
    # Process each dataset
    processed_data = {}
    for name, df in data.items():
        processor = DataProcessor(df)
        
        # Clean data
        cleaned_data = processor.clean_data()
        
        # Create dimension tables if this is the main dataset
        if name == 'listings':
            dimensions = processor.create_dimension_tables()
            fact_table = processor.create_fact_table(dimensions)
            
            # Save dimension and fact tables
            processor.save_to_csv(
                {**dimensions, 'fact_listings': fact_table},
                'data/processed'
            )
            
            # Upload to GCS if credentials are available
            if os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
                for table_name, table_df in dimensions.items():
                    processor.upload_to_gcs(
                        'your-bucket-name',
                        f'data/processed/{table_name}.csv',
                        f'processed/{table_name}.csv'
                    )
        
        processed_data[name] = cleaned_data
    
    logger.info("Phase 2 completed successfully")
    return processed_data

def run_phase3(data: pd.DataFrame):
    """Run Phase 3: Feature Engineering"""
    logger.info("Starting Phase 3: Feature Engineering")
    
    engineer = FeatureEngineer(data)
    
    # Create various types of features
    engineer.create_temporal_features()
    engineer.create_price_features()
    engineer.create_location_features()
    engineer.create_text_features()
    
    # Scale and encode features
    engineer.scale_numerical_features()
    engineer.encode_categorical_features()
    
    # Save engineered features
    engineer.save_features('data/interim/engineered_features.csv')
    
    logger.info("Phase 3 completed successfully")
    return engineer.data

def run_phase4(data: pd.DataFrame):
    """Run Phase 4: Model Training"""
    logger.info("Starting Phase 4: Model Training")
    
    trainer = ModelTrainer(data)
    
    # Prepare data for price prediction
    price_features = [col for col in data.columns if col != 'price_x']
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        'price_x', price_features)
    
    # Train and evaluate price prediction model
    trainer.train_price_prediction_model(X_train, y_train)
    metrics = trainer.evaluate_regression_model(
        'price_prediction', X_test, y_test)
    
    # Cross-validate model
    cv_scores = trainer.cross_validate_model(
        'price_prediction', X_train, y_train)
    
    # Save models and results
    trainer.save_models('models')
    
    logger.info("Phase 4 completed successfully")
    return trainer

def main():
    """Main function to run all phases of the project."""
    logger.info("Starting Airbnb Data Analysis Pipeline")
    
    # Create directory structure
    create_directory_structure()
    
    # Define data paths
    data_paths = {
        'calendar': 'data/raw/calendar.csv',
        'listings': 'data/raw/listings.csv',
        'reviews': 'data/raw/reviews.csv'
    }
    
    # Run all phases
    raw_data = run_phase1(data_paths)
    processed_data = run_phase2(raw_data)
    engineered_data = run_phase3(processed_data['listings'])
    model = run_phase4(engineered_data)
    
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    main()
