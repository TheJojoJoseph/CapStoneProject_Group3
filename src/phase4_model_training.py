import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import logging
from typing import Dict, List, Tuple, Union
import joblib
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize ModelTrainer with input DataFrame.
        
        Args:
            data (pd.DataFrame): Input DataFrame for model training
        """
        self.data = data
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
    
    def prepare_data(self, target_col: str, feature_cols: List[str],
                    test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray,
                                                    np.ndarray, np.ndarray]:
        """Prepare data for model training."""
        logger.info("Preparing data for model training...")
        
        # Select features and target
        X = self.data[feature_cols]
        y = self.data[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler for later use
        self.scalers['feature_scaler'] = scaler
        
        logger.info("Data preparation completed")
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_price_prediction_model(self, X_train: np.ndarray, y_train: np.ndarray,
                                   model_type: str = 'random_forest') -> None:
        """Train a model for price prediction."""
        logger.info(f"Training {model_type} model for price prediction...")
        
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError("Invalid model type. Use 'linear' or 'random_forest'")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Store model
        self.models['price_prediction'] = model
        
        # Calculate feature importance if available
        if hasattr(model, 'feature_importances_'):
            self.feature_importance['price_prediction'] = model.feature_importances_
        
        logger.info("Price prediction model training completed")
    
    def train_booking_probability_model(self, X_train: np.ndarray,
                                      y_train: np.ndarray,
                                      model_type: str = 'random_forest') -> None:
        """Train a model for booking probability prediction."""
        logger.info(f"Training {model_type} model for booking probability...")
        
        if model_type == 'logistic':
            model = LogisticRegression(random_state=42)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError("Invalid model type. Use 'logistic' or 'random_forest'")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Store model
        self.models['booking_probability'] = model
        
        # Calculate feature importance if available
        if hasattr(model, 'feature_importances_'):
            self.feature_importance['booking_probability'] = model.feature_importances_
        
        logger.info("Booking probability model training completed")
    
    def evaluate_regression_model(self, model_name: str, X_test: np.ndarray,
                                y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate regression model performance."""
        logger.info(f"Evaluating {model_name} model...")
        
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        logger.info(f"Model evaluation metrics: {metrics}")
        return metrics
    
    def evaluate_classification_model(self, model_name: str, X_test: np.ndarray,
                                    y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate classification model performance."""
        logger.info(f"Evaluating {model_name} model...")
        
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        logger.info(f"Model evaluation metrics: {metrics}")
        return metrics
    
    def cross_validate_model(self, model_name: str, X: np.ndarray, y: np.ndarray,
                           cv: int = 5) -> List[float]:
        """Perform cross-validation."""
        logger.info(f"Performing {cv}-fold cross-validation for {model_name}...")
        
        model = self.models[model_name]
        scores = cross_val_score(model, X, y, cv=cv)
        
        logger.info(f"Cross-validation scores: {scores}")
        logger.info(f"Mean CV score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        
        return scores
    
    def save_models(self, output_dir: str) -> None:
        """Save trained models and scalers."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            model_path = output_path / f"{name}_model.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Saved {name} model to {model_path}")
        
        # Save scalers
        for name, scaler in self.scalers.items():
            scaler_path = output_path / f"{name}.joblib"
            joblib.dump(scaler, scaler_path)
            logger.info(f"Saved {name} to {scaler_path}")
        
        # Save feature importance
        for name, importance in self.feature_importance.items():
            importance_path = output_path / f"{name}_feature_importance.csv"
            pd.DataFrame({'importance': importance}).to_csv(importance_path)
            logger.info(f"Saved {name} feature importance to {importance_path}")
    
    def load_model(self, model_path: str) -> None:
        """Load a trained model."""
        model = joblib.load(model_path)
        model_name = Path(model_path).stem.replace('_model', '')
        self.models[model_name] = model
        logger.info(f"Loaded model from {model_path}")
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Make predictions using a trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Scale features if scaler exists
        if 'feature_scaler' in self.scalers:
            X = self.scalers['feature_scaler'].transform(X)
        
        return self.models[model_name].predict(X)
