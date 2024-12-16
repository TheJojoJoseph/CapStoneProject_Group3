import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import bigquery
import os

class AirbnbVisualizer:
    def __init__(self, project_id):
        """Initialize the visualizer with GCP project ID."""
        self.client = bigquery.Client(project=project_id)
        self.setup_style()
    
    def setup_style(self):
        """Set up the visualization style."""
        plt.style.use('default')  # Using default style instead of seaborn
        sns.set_theme()  # This will apply seaborn styling
    
    def query_data(self, query):
        """Execute a BigQuery query and return results as a DataFrame."""
        return self.client.query(query).to_dataframe()
    
    def save_plot(self, plt, filename, output_dir='visualizations'):
        """Save the plot to the specified directory."""
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
        plt.close()

    def price_distribution(self):
        """Create a price distribution visualization."""
        query = """
        SELECT price
        FROM `airbnb_analytics.listings`
        WHERE price > 0 AND price < 1000
        """
        df = self.query_data(query)
        
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x='price', bins=50)
        plt.title('Distribution of Airbnb Prices')
        plt.xlabel('Price ($)')
        plt.ylabel('Count')
        
        self.save_plot(plt, 'price_distribution.png')

    def room_type_analysis(self):
        """Create room type analysis visualization."""
        query = """
        SELECT room_type, COUNT(*) as count, AVG(price) as avg_price
        FROM `airbnb_analytics.listings`
        GROUP BY room_type
        ORDER BY count DESC
        """
        df = self.query_data(query)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Room type distribution
        sns.barplot(data=df, x='room_type', y='count', ax=ax1)
        ax1.set_title('Distribution of Room Types')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        
        # Average price by room type
        sns.barplot(data=df, x='room_type', y='avg_price', ax=ax2)
        ax2.set_title('Average Price by Room Type')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        self.save_plot(plt, 'room_type_analysis.png')

    def location_heatmap(self):
        """Create a location heatmap visualization."""
        query = """
        SELECT latitude, longitude, price
        FROM `airbnb_analytics.listings`
        WHERE price > 0 AND price < 1000
        """
        df = self.query_data(query)
        
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df, x='longitude', y='latitude', hue='price', size='price',
                       sizes=(20, 200), palette='viridis')
        plt.title('Price Distribution by Location')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        self.save_plot(plt, 'location_heatmap.png')

    def amenities_analysis(self):
        """Create amenities analysis visualization."""
        query = """
        SELECT amenities, AVG(price) as avg_price, COUNT(*) as count
        FROM `airbnb_analytics.listings`
        GROUP BY amenities
        ORDER BY count DESC
        LIMIT 10
        """
        df = self.query_data(query)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='amenities', y='avg_price')
        plt.title('Average Price by Top 10 Amenity Combinations')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        self.save_plot(plt, 'amenities_analysis.png')

    def reviews_analysis(self):
        """Create reviews analysis visualization."""
        query = """
        SELECT 
            review_scores_rating,
            price,
            number_of_reviews
        FROM `airbnb_analytics.listings`
        WHERE review_scores_rating IS NOT NULL
            AND price > 0 AND price < 1000
        """
        df = self.query_data(query)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Rating vs Price
        sns.scatterplot(data=df, x='review_scores_rating', y='price', ax=ax1, alpha=0.5)
        ax1.set_title('Rating vs Price')
        
        # Number of Reviews Distribution
        sns.histplot(data=df, x='number_of_reviews', bins=50, ax=ax2)
        ax2.set_title('Distribution of Number of Reviews')
        
        plt.tight_layout()
        self.save_plot(plt, 'reviews_analysis.png')
