import functions_framework
from google.cloud import bigquery
from google.cloud import storage
from google.cloud import aiplatform
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime
import logging
import joblib
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
bq_client = bigquery.Client()
storage_client = storage.Client()

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal features from date columns."""
    date_columns = ['host_since', 'first_review', 'last_review']
    
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df[f'days_since_{col}'] = (datetime.now() - df[col]).dt.days
    
    return df

def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create price-related features."""
    if 'price_x' in df.columns:
        df['price_x'] = df['price_x'].replace('[\$,]', '', regex=True).astype(float)
        df['price_bucket'] = pd.qcut(df['price_x'], q=5, labels=['very_low', 'low',
                                                                'medium', 'high',
                                                                'very_high'])
        if 'accommodates' in df.columns:
            df['price_per_person'] = df['price_x'] / df['accommodates']
    
    return df

def create_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create location-based features."""
    if all(col in df.columns for col in ['latitude', 'longitude']):
        # Example coordinates (Seattle city center)
        city_center = {'lat': 47.6062, 'lon': -122.3321}
        
        df['distance_to_center'] = np.sqrt(
            (df['latitude'] - city_center['lat'])**2 +
            (df['longitude'] - city_center['lon'])**2
        )
    
    return df

def scale_and_encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale numerical features and encode categorical features."""
    # Scale numerical features
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Encode categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    encoders = {}
    
    for col in categorical_cols:
        encoder = LabelEncoder()
        df[f'{col}_encoded'] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder
    
    # Save encoders to GCS
    bucket = storage_client.bucket(os.environ['MODEL_BUCKET'])
    for name, encoder in encoders.items():
        blob = bucket.blob(f'encoders/{name}_encoder.joblib')
        joblib.dump(encoder, f'/tmp/{name}_encoder.joblib')
        blob.upload_from_filename(f'/tmp/{name}_encoder.joblib')
    
    return df

@functions_framework.cloud_event
def engineer_features(cloud_event):
    """Engineer features when triggered."""
    try:
        # Query processed data from BigQuery
        query = """
        SELECT l.*, h.*, g.*
        FROM `airbnb_analytics.fact_listings` l
        JOIN `airbnb_analytics.dim_host` h ON l.host_id = h.host_id
        JOIN `airbnb_analytics.dim_geography` g ON l.listing_id = g.listing_id
        """
        df = bq_client.query(query).to_dataframe()
        
        # Create features
        df = create_temporal_features(df)
        df = create_price_features(df)
        df = create_location_features(df)
        df = scale_and_encode_features(df)
        
        # Save engineered features to BigQuery
        table_id = f"{bq_client.project}.airbnb_analytics.engineered_features"
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        )
        
        job = bq_client.load_table_from_dataframe(
            df, table_id, job_config=job_config
        )
        job.result()
        
        logger.info("Feature engineering completed successfully")
        
    except Exception as e:
        logger.error(f"Error engineering features: {str(e)}")
        raise
