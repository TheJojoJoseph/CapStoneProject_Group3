import pandas as pd
import numpy as np
from google.cloud import bigquery
import os

def generate_sample_data(num_records=1000):
    """Generate sample Airbnb listing data."""
    np.random.seed(42)
    
    data = {
        'price': np.random.normal(100, 50, num_records),
        'room_type': np.random.choice(['Entire home/apt', 'Private room', 'Shared room'], num_records),
        'latitude': np.random.uniform(37.7, 37.8, num_records),
        'longitude': np.random.uniform(-122.5, -122.4, num_records),
        'amenities': np.random.choice(['Wifi,Kitchen', 'Wifi,Kitchen,Parking', 'Wifi,Pool', 'Kitchen,Parking'], num_records),
        'review_scores_rating': np.random.uniform(3.5, 5.0, num_records),
        'number_of_reviews': np.random.poisson(10, num_records)
    }
    
    return pd.DataFrame(data)

def upload_to_bigquery(project_id, dataset_name='airbnb_analytics', table_name='listings'):
    """Upload sample data to BigQuery."""
    # Generate sample data
    df = generate_sample_data()
    
    # Initialize BigQuery client
    client = bigquery.Client(project=project_id)
    
    # Create dataset if it doesn't exist
    dataset_ref = f"{project_id}.{dataset_name}"
    try:
        client.get_dataset(dataset_ref)
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "us-central1"
        client.create_dataset(dataset, exists_ok=True)
    
    # Upload data to BigQuery
    table_ref = f"{dataset_ref}.{table_name}"
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
    )
    
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()  # Wait for the job to complete
    
    print(f"Loaded {len(df)} rows into {table_ref}")

if __name__ == "__main__":
    project_id = "lateral-vision-438701-u5"
    upload_to_bigquery(project_id)
