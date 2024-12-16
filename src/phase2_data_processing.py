import pandas as pd
import numpy as np
from google.cloud import storage
from typing import Dict, List, Optional
import logging
from pathlib import Path
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, input_data: pd.DataFrame):
        """
        Initialize DataProcessor with input DataFrame.
        
        Args:
            input_data (pd.DataFrame): Input DataFrame to process
        """
        self.data = input_data
        self.processed_data = None
    
    def clean_data(self) -> pd.DataFrame:
        """Clean the data by handling missing values and outliers."""
        logger.info("Starting data cleaning process...")
        
        # Create a copy of the data
        df = self.data.copy()
        
        # Handle missing values
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Fill numerical missing values with median
        for col in numerical_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical missing values with mode
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        # Remove outliers using IQR method for numerical columns
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
        
        self.processed_data = df
        logger.info(f"Data cleaning completed. Shape after cleaning: {df.shape}")
        return df
    
    def create_dimension_tables(self) -> Dict[str, pd.DataFrame]:
        """Create dimension tables for the star schema."""
        logger.info("Creating dimension tables...")
        
        # Create date dimension
        def create_date_dimension(df: pd.DataFrame) -> pd.DataFrame:
            dates = pd.concat([
                pd.to_datetime(df['first_review']),
                pd.to_datetime(df['last_review'])
            ]).unique()
            
            date_dim = pd.DataFrame({
                'date_id': range(len(dates)),
                'date': dates,
                'year': pd.DatetimeIndex(dates).year,
                'month': pd.DatetimeIndex(dates).month,
                'day': pd.DatetimeIndex(dates).day,
                'day_of_week': pd.DatetimeIndex(dates).dayofweek,
                'is_weekend': pd.DatetimeIndex(dates).dayofweek.isin([5, 6])
            })
            return date_dim
        
        # Create geography dimension
        def create_geography_dimension(df: pd.DataFrame) -> pd.DataFrame:
            geo_dim = df[['neighbourhood_cleansed', 'city', 'state', 'country',
                         'latitude', 'longitude']].drop_duplicates()
            geo_dim['geography_id'] = range(len(geo_dim))
            return geo_dim
        
        # Create host dimension
        def create_host_dimension(df: pd.DataFrame) -> pd.DataFrame:
            host_dim = df[['host_id', 'host_name', 'host_since', 'host_location']].drop_duplicates()
            return host_dim
        
        # Create property dimension
        def create_property_dimension(df: pd.DataFrame) -> pd.DataFrame:
            property_dim = df[['listing_id', 'name', 'property_type', 'room_type',
                             'accommodates', 'bathrooms', 'bedrooms', 'beds']].drop_duplicates()
            return property_dim
        
        dimensions = {
            'dim_date': create_date_dimension(self.processed_data),
            'dim_geography': create_geography_dimension(self.processed_data),
            'dim_host': create_host_dimension(self.processed_data),
            'dim_property': create_property_dimension(self.processed_data)
        }
        
        logger.info("Dimension tables created successfully")
        return dimensions
    
    def create_fact_table(self, dimensions: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create the fact table with foreign keys to dimension tables."""
        logger.info("Creating fact table...")
        
        fact_table = self.processed_data[['listing_id', 'host_id', 'price_x',
                                        'review_scores_rating', 'reviews_per_month',
                                        'number_of_reviews', 'availability_30',
                                        'availability_60', 'availability_90',
                                        'availability_365']]
        
        # Add foreign keys from dimensions
        fact_table = fact_table.merge(dimensions['dim_property'][['listing_id']],
                                    on='listing_id', how='left')
        fact_table = fact_table.merge(dimensions['dim_host'][['host_id']],
                                    on='host_id', how='left')
        
        logger.info("Fact table created successfully")
        return fact_table
    
    def save_to_csv(self, data: Dict[str, pd.DataFrame], output_dir: str) -> None:
        """Save processed dataframes to CSV files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, df in data.items():
            file_path = output_path / f"{name}.csv"
            df.to_csv(file_path, index=False)
            logger.info(f"Saved {name} to {file_path}")
    
    def upload_to_gcs(self, bucket_name: str, local_file_path: str,
                     destination_blob_name: str) -> None:
        """Upload a file to Google Cloud Storage."""
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(destination_blob_name)
            
            blob.upload_from_filename(local_file_path)
            logger.info(f"File {local_file_path} uploaded to {destination_blob_name}")
            
        except Exception as e:
            logger.error(f"Error uploading to GCS: {str(e)}")
            raise
