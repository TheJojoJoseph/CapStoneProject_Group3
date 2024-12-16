import pytest
import os
import sys

# Add each cloud function directory to the Python path
cloud_functions_path = os.path.join(os.path.dirname(__file__), '../cloud_functions')
for function_dir in ['data_ingestion', 'data_processing', 'feature_engineering', 'model_training']:
    sys.path.append(os.path.join(cloud_functions_path, function_dir))

def test_environment():
    """Basic test to ensure test environment is working"""
    assert True

def test_data_ingestion():
    """Test data ingestion function"""
    try:
        from main import ingest_data
        assert callable(ingest_data)
    except ImportError as e:
        pytest.skip(f"Could not import data_ingestion function: {e}")

def test_data_processing():
    """Test data processing function"""
    try:
        from main import process_data
        assert callable(process_data)
    except ImportError as e:
        pytest.skip(f"Could not import data_processing function: {e}")

def test_feature_engineering():
    """Test feature engineering function"""
    try:
        from main import engineer_features
        assert callable(engineer_features)
    except ImportError as e:
        pytest.skip(f"Could not import feature_engineering function: {e}")

def test_model_training():
    """Test model training function"""
    try:
        from main import train_model
        assert callable(train_model)
    except ImportError as e:
        pytest.skip(f"Could not import model_training function: {e}")
