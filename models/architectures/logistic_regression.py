"""
Logistic Regression Model

Baseline logistic regression classifier for predicting home team win probability.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import logging

from models.base import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogisticRegressionModel(BaseModel):
    """
    Logistic regression classifier with L2 regularization.
    
    Uses StandardScaler to normalize features before training.
    """
    
    def __init__(self, C=1.0, max_iter=1000, random_state=42):
        """
        Initialize logistic regression model.
        
        Args:
            C: Inverse of regularization strength (smaller = stronger regularization)
            max_iter: Maximum iterations for solver
            random_state: Random seed
        """
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            solver="lbfgs",  # Good for small-medium datasets
            penalty="l2",
        )
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def fit(self, X, y):
        """
        Train the logistic regression model.
        
        Args:
            X: Feature matrix (n_samples, n_features) or DataFrame
            y: Target vector (n_samples,) - binary (1 = home win, 0 = away win)
        """
        # Store feature names if DataFrame
        if hasattr(X, "columns"):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        logger.info(f"Training logistic regression on {len(X)} samples with {X.shape[1]} features")
        self.model.fit(X_scaled, y)
        
        logger.info(f"Training complete. Intercept: {self.model.intercept_[0]:.4f}")
    
    def predict_proba(self, X):
        """
        Predict home team win probabilities.
        
        Args:
            X: Feature matrix (n_samples, n_features) or DataFrame
        
        Returns:
            Array of shape (n_samples,) with home win probabilities
        """
        # Convert to array if DataFrame
        if hasattr(X, "values"):
            X = X.values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict probabilities (returns [P(0), P(1)], we want P(1) = home win)
        proba = self.model.predict_proba(X_scaled)
        return proba[:, 1]  # Return probability of class 1 (home win)
    
    def get_feature_importance(self):
        """
        Get feature importance (coefficients).
        
        Returns:
            Dictionary mapping feature names to coefficients
        """
        if self.feature_names is None:
            return None
        
        coef = self.model.coef_[0]
        return dict(zip(self.feature_names, coef))

