import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Tuple, Union
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize FeatureEngineer with input DataFrame.
        
        Args:
            data (pd.DataFrame): Input DataFrame for feature engineering
        """
        self.data = data.copy()
        self.scalers = {}
        self.encoders = {}
        
    def create_temporal_features(self) -> pd.DataFrame:
        """Create temporal features from date columns."""
        logger.info("Creating temporal features...")
        
        date_columns = ['host_since', 'first_review', 'last_review']
        
        for col in date_columns:
            if col in self.data.columns:
                self.data[col] = pd.to_datetime(self.data[col])
                
                # Extract basic date components
                self.data[f'{col}_year'] = self.data[col].dt.year
                self.data[f'{col}_month'] = self.data[col].dt.month
                self.data[f'{col}_day'] = self.data[col].dt.day
                self.data[f'{col}_dayofweek'] = self.data[col].dt.dayofweek
                
                # Calculate time since date
                self.data[f'days_since_{col}'] = (
                    datetime.now() - self.data[col]).dt.days
        
        logger.info("Temporal features created successfully")
        return self.data
    
    def create_price_features(self) -> pd.DataFrame:
        """Create features related to pricing."""
        logger.info("Creating price features...")
        
        if 'price_x' in self.data.columns:
            # Clean price column (remove $ and convert to float)
            self.data['price_x'] = self.data['price_x'].replace('[\$,]', '', 
                                                               regex=True).astype(float)
            
            # Create price buckets
            self.data['price_bucket'] = pd.qcut(self.data['price_x'], 
                                              q=5, labels=['very_low', 'low', 
                                                         'medium', 'high', 'very_high'])
            
            # Calculate price per accommodation
            if 'accommodates' in self.data.columns:
                self.data['price_per_person'] = (self.data['price_x'] / 
                                               self.data['accommodates'])
        
        logger.info("Price features created successfully")
        return self.data
    
    def create_location_features(self) -> pd.DataFrame:
        """Create features based on location data."""
        logger.info("Creating location features...")
        
        if all(col in self.data.columns for col in ['latitude', 'longitude']):
            # Calculate distance from city center (example coordinates for demonstration)
            city_center = {'lat': 47.6062, 'lon': -122.3321}  # Seattle coordinates
            
            self.data['distance_to_center'] = np.sqrt(
                (self.data['latitude'] - city_center['lat'])**2 +
                (self.data['longitude'] - city_center['lon'])**2
            )
            
            # Create location clusters (if sklearn is available)
            try:
                from sklearn.cluster import KMeans
                coords = self.data[['latitude', 'longitude']].values
                kmeans = KMeans(n_clusters=5, random_state=42)
                self.data['location_cluster'] = kmeans.fit_predict(coords)
            except ImportError:
                logger.warning("sklearn not available for location clustering")
        
        logger.info("Location features created successfully")
        return self.data
    
    def create_text_features(self) -> pd.DataFrame:
        """Create features from text columns using TF-IDF."""
        logger.info("Creating text features...")
        
        text_columns = ['name', 'description'] # Add more text columns as needed
        
        for col in text_columns:
            if col in self.data.columns:
                # Initialize TF-IDF vectorizer
                tfidf = TfidfVectorizer(max_features=10, stop_words='english')
                
                # Fill NaN values with empty string
                text_data = self.data[col].fillna('')
                
                # Generate TF-IDF features
                tfidf_features = tfidf.fit_transform(text_data)
                
                # Convert to DataFrame
                feature_names = [f'{col}_tfidf_{i}' for i in range(tfidf_features.shape[1])]
                tfidf_df = pd.DataFrame(tfidf_features.toarray(), 
                                      columns=feature_names,
                                      index=self.data.index)
                
                # Concatenate with original DataFrame
                self.data = pd.concat([self.data, tfidf_df], axis=1)
        
        logger.info("Text features created successfully")
        return self.data
    
    def scale_numerical_features(self, method: str = 'standard') -> pd.DataFrame:
        """Scale numerical features using specified method."""
        logger.info(f"Scaling numerical features using {method} method...")
        
        numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        
        for col in numerical_cols:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError("Invalid scaling method. Use 'standard' or 'minmax'")
            
            self.data[f'{col}_scaled'] = scaler.fit_transform(
                self.data[col].values.reshape(-1, 1))
            self.scalers[col] = scaler
        
        logger.info("Numerical features scaled successfully")
        return self.data
    
    def encode_categorical_features(self, method: str = 'label') -> pd.DataFrame:
        """Encode categorical features using specified method."""
        logger.info(f"Encoding categorical features using {method} method...")
        
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if method == 'label':
                encoder = LabelEncoder()
                self.data[f'{col}_encoded'] = encoder.fit_transform(
                    self.data[col].astype(str))
                self.encoders[col] = encoder
            elif method == 'onehot':
                # Create dummy variables
                dummies = pd.get_dummies(self.data[col], prefix=col)
                self.data = pd.concat([self.data, dummies], axis=1)
            else:
                raise ValueError("Invalid encoding method. Use 'label' or 'onehot'")
        
        logger.info("Categorical features encoded successfully")
        return self.data
    
    def save_features(self, output_path: str) -> None:
        """Save engineered features to CSV."""
        self.data.to_csv(output_path, index=False)
        logger.info(f"Engineered features saved to {output_path}")
    
    def get_feature_importance(self, target_col: str) -> pd.DataFrame:
        """Calculate feature importance using correlation with target."""
        correlations = self.data.corr()[target_col].sort_values(ascending=False)
        return pd.DataFrame({'feature': correlations.index,
                           'importance': correlations.values})
