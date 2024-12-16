import functions_framework
from google.cloud import bigquery
from google.cloud import storage
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
bq_client = bigquery.Client()
storage_client = storage.Client()

def create_dimension_tables(df: pd.DataFrame) -> dict:
    """Create dimension tables from the main dataset."""
    dimensions = {}
    
    # Create date dimension
    date_columns = ['host_since', 'first_review', 'last_review']
    dates = pd.concat([pd.to_datetime(df[col]) for col in date_columns 
                      if col in df.columns]).unique()
    
    dimensions['dim_date'] = pd.DataFrame({
        'date_id': range(len(dates)),
        'date': dates,
        'year': pd.DatetimeIndex(dates).year,
        'month': pd.DatetimeIndex(dates).month,
        'day': pd.DatetimeIndex(dates).day,
        'day_of_week': pd.DatetimeIndex(dates).dayofweek
    })
    
    # Create geography dimension
    dimensions['dim_geography'] = df[['neighbourhood_cleansed', 'city', 'state',
                                    'country', 'latitude', 'longitude']].drop_duplicates()
    dimensions['dim_geography']['geography_id'] = range(len(dimensions['dim_geography']))
    
    # Create host dimension
    dimensions['dim_host'] = df[['host_id', 'host_name', 'host_since',
                                'host_location']].drop_duplicates()
    
    # Create property dimension
    dimensions['dim_property'] = df[['listing_id', 'name', 'property_type',
                                   'room_type', 'accommodates', 'bathrooms',
                                   'bedrooms', 'beds']].drop_duplicates()
    
    return dimensions

def load_to_bigquery(df: pd.DataFrame, table_name: str, dataset: str = 'airbnb_analytics') -> None:
    """Load DataFrame to BigQuery table."""
    table_id = f"{bq_client.project}.{dataset}.{table_name}"
    
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    
    job = bq_client.load_table_from_dataframe(
        df, table_id, job_config=job_config
    )
    job.result()
    
    logger.info(f"Loaded {len(df)} rows into {table_id}")

@functions_framework.cloud_event
def process_data(cloud_event):
    """Process data and create dimension tables when triggered."""
    try:
        # Query raw data from BigQuery
        query = """
        SELECT *
        FROM `airbnb_analytics.listings`
        """
        df = bq_client.query(query).to_dataframe()
        
        # Create dimension tables
        dimensions = create_dimension_tables(df)
        
        # Create fact table
        fact_table = df[['listing_id', 'host_id', 'price_x', 'review_scores_rating',
                        'reviews_per_month', 'number_of_reviews', 'availability_30',
                        'availability_60', 'availability_90', 'availability_365']]
        
        # Load dimension tables to BigQuery
        for table_name, table_df in dimensions.items():
            load_to_bigquery(table_df, table_name)
        
        # Load fact table
        load_to_bigquery(fact_table, 'fact_listings')
        
        logger.info("Data processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise
