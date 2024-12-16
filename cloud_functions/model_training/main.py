import functions_framework
from google.cloud import bigquery
from google.cloud import storage
from google.cloud import aiplatform
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import logging
import joblib
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
bq_client = bigquery.Client()
storage_client = storage.Client()

def prepare_training_data() -> tuple:
    """Prepare data for model training."""
    query = """
    SELECT *
    FROM `airbnb_analytics.engineered_features`
    """
    df = bq_client.query(query).to_dataframe()
    
    # Select features and target
    target_col = 'price_x'
    feature_cols = [col for col in df.columns if col != target_col]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """Train the price prediction model."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: RandomForestRegressor, X_test: pd.DataFrame,
                  y_test: pd.Series) -> dict:
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred)
    }
    
    return metrics

def save_model_to_gcs(model: RandomForestRegressor, metrics: dict) -> None:
    """Save model and metrics to Google Cloud Storage."""
    bucket = storage_client.bucket(os.environ['MODEL_BUCKET'])
    
    # Save model
    model_path = '/tmp/model.joblib'
    joblib.dump(model, model_path)
    blob = bucket.blob('models/price_prediction_model.joblib')
    blob.upload_from_filename(model_path)
    
    # Save metrics
    metrics_path = '/tmp/metrics.json'
    pd.DataFrame([metrics]).to_json(metrics_path)
    blob = bucket.blob('metrics/model_metrics.json')
    blob.upload_from_filename(metrics_path)

def deploy_model_to_vertex(model_path: str, model_name: str) -> None:
    """Deploy model to Vertex AI."""
    aiplatform.init(project=os.environ['PROJECT_ID'])
    
    # Upload model to Vertex AI
    model = aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=f"gs://{os.environ['MODEL_BUCKET']}/models",
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-23:latest"
    )
    
    # Deploy model
    endpoint = model.deploy(
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=5
    )
    
    logger.info(f"Model deployed to endpoint: {endpoint.resource_name}")

@functions_framework.cloud_event
def train_and_deploy_model(cloud_event):
    """Train and deploy model when triggered."""
    try:
        # Prepare data
        X_train, X_test, y_train, y_test = prepare_training_data()
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        logger.info(f"Model metrics: {metrics}")
        
        # Save model and metrics
        save_model_to_gcs(model, metrics)
        
        # Deploy to Vertex AI
        model_path = f"gs://{os.environ['MODEL_BUCKET']}/models/price_prediction_model.joblib"
        deploy_model_to_vertex(model_path, "airbnb_price_predictor")
        
        logger.info("Model training and deployment completed successfully")
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise
