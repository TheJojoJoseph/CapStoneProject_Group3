import functions_framework
from google.cloud import storage
from google.cloud import bigquery
import pandas as pd
import logging
import os
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
storage_client = storage.Client()
bq_client = bigquery.Client()

@functions_framework.cloud_event
def ingest_data(cloud_event):
    """Triggered by a change to a Cloud Storage bucket."""
    data = cloud_event.data

    bucket = data["bucket"]
    name = data["name"]
    
    logger.info(f"Processing file: {name} in bucket: {bucket}")

    # Download file from Cloud Storage
    bucket = storage_client.bucket(bucket)
    blob = bucket.blob(name)
    
    # Create temporary file
    temp_file = f"/tmp/{name}"
    blob.download_to_filename(temp_file)
    
    try:
        # Read data
        df = pd.read_csv(temp_file)
        
        # Determine table name from filename
        table_name = os.path.splitext(name)[0]
        
        # Load to BigQuery
        dataset_ref = bq_client.dataset('airbnb_analytics')
        table_ref = dataset_ref.table(table_name)
        
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            source_format=bigquery.SourceFormat.CSV,
            autodetect=True
        )
        
        with open(temp_file, "rb") as source_file:
            job = bq_client.load_table_from_file(
                source_file,
                table_ref,
                job_config=job_config
            )
        
        job.result()  # Wait for the job to complete
        
        logger.info(f"Loaded {job.output_rows} rows into {table_ref}")
        
    except Exception as e:
        logger.error(f"Error processing {name}: {str(e)}")
        raise
        
    finally:
        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)
