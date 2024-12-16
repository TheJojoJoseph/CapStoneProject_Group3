"""GCP Configuration Settings"""

# GCP Project settings
PROJECT_ID = 'your-project-id'  # Replace with your GCP project ID
REGION = 'us-central1'          # Replace with your preferred region

# Cloud Storage settings
RAW_DATA_BUCKET = f'{PROJECT_ID}-raw-data'
PROCESSED_DATA_BUCKET = f'{PROJECT_ID}-processed-data'
MODEL_BUCKET = f'{PROJECT_ID}-models'

# BigQuery settings
DATASET_ID = 'airbnb_analytics'
TABLES = {
    'dim_date': 'dim_date',
    'dim_geography': 'dim_geography',
    'dim_host': 'dim_host',
    'dim_property': 'dim_property',
    'fact_listings': 'fact_listings'
}

# Pub/Sub settings
TOPIC_ID = 'data-pipeline-events'

# Cloud Functions settings
FUNCTION_REGION = REGION
FUNCTION_MEMORY = '2048MB'
FUNCTION_TIMEOUT = '540s'

# AI Platform settings
MODEL_NAME = 'airbnb_price_predictor'
MODEL_VERSION = 'v1'
FRAMEWORK = 'SCIKIT_LEARN'

# Monitoring settings
MONITORING_WORKSPACE = 'airbnb-monitoring'
