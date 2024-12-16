import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataExplorer:
    def __init__(self, data_paths: Dict[str, str]):
        """
        Initialize DataExplorer with paths to required datasets.
        
        Args:
            data_paths (Dict[str, str]): Dictionary containing paths to required CSV files
                                       (calendar, listings, reviews)
        """
        self.data_paths = data_paths
        self.datasets = {}
        
    def load_data(self) -> None:
        """Load all datasets into memory."""
        try:
            for name, path in self.data_paths.items():
                logger.info(f"Loading {name} dataset from {path}")
                self.datasets[name] = pd.read_csv(path)
                logger.info(f"Successfully loaded {name} dataset with shape {self.datasets[name].shape}")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def analyze_missing_values(self) -> Dict[str, pd.Series]:
        """Analyze missing values in all datasets."""
        missing_values = {}
        for name, df in self.datasets.items():
            missing_values[name] = df.isnull().sum()
            logger.info(f"\nMissing Values in {name}:\n{missing_values[name]}")
        return missing_values
    
    def generate_summary_statistics(self) -> Dict[str, pd.DataFrame]:
        """Generate summary statistics for all datasets."""
        summary_stats = {}
        for name, df in self.datasets.items():
            summary_stats[name] = df.describe()
            logger.info(f"\nSummary Statistics for {name}:\n{summary_stats[name]}")
        return summary_stats
    
    def plot_missing_values(self, save_path: str = None) -> None:
        """Create visualizations for missing values."""
        for name, df in self.datasets.items():
            plt.figure(figsize=(15, 5))
            sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
            plt.title(f'Missing Values in {name} Dataset')
            if save_path:
                plt.savefig(f"{save_path}/missing_values_{name}.png")
            plt.close()
    
    def analyze_numerical_distributions(self, save_path: str = None) -> None:
        """Analyze and plot distributions of numerical columns."""
        for name, df in self.datasets.items():
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
            for col in numerical_cols:
                plt.figure(figsize=(10, 5))
                sns.histplot(data=df, x=col, kde=True)
                plt.title(f'Distribution of {col} in {name}')
                if save_path:
                    plt.savefig(f"{save_path}/distribution_{name}_{col}.png")
                plt.close()
    
    def generate_correlation_matrix(self, save_path: str = None) -> Dict[str, pd.DataFrame]:
        """Generate correlation matrices for numerical columns."""
        correlation_matrices = {}
        for name, df in self.datasets.items():
            numerical_df = df.select_dtypes(include=['int64', 'float64'])
            correlation_matrices[name] = numerical_df.corr()
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(correlation_matrices[name], annot=True, cmap='coolwarm', center=0)
            plt.title(f'Correlation Matrix for {name}')
            if save_path:
                plt.savefig(f"{save_path}/correlation_matrix_{name}.png")
            plt.close()
        
        return correlation_matrices
    
    def save_analysis_report(self, output_path: str) -> None:
        """Save a comprehensive analysis report."""
        with open(output_path, 'w') as f:
            f.write("Data Analysis Report\n")
            f.write("===================\n\n")
            
            for name, df in self.datasets.items():
                f.write(f"\n{name.upper()} Dataset Analysis\n")
                f.write("-" * (len(name) + 18) + "\n")
                f.write(f"Shape: {df.shape}\n")
                f.write(f"Columns: {', '.join(df.columns)}\n")
                f.write("\nMissing Values:\n")
                f.write(df.isnull().sum().to_string())
                f.write("\n\nSummary Statistics:\n")
                f.write(df.describe().to_string())
                f.write("\n" + "="*50 + "\n")
