"""
Gradient Boosting Model

Gradient boosting classifier for predicting home team win probability.
Uses XGBoost if available, falls back to sklearn GradientBoostingClassifier.
"""

import numpy as np
import logging

from models.base import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import XGBoost, fall back to sklearn
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    from sklearn.ensemble import GradientBoostingClassifier
    logger.warning("XGBoost not available, using sklearn GradientBoostingClassifier")


class GradientBoostingModel(BaseModel):
    """
    Gradient boosting classifier for home team win prediction.
    
    Uses XGBoost if available, otherwise sklearn GradientBoostingClassifier.
    """
    
    def __init__(
        self,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        **kwargs
    ):
        """
        Initialize gradient boosting model.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate (shrinkage)
            random_state: Random seed
            **kwargs: Additional parameters passed to underlying model
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.kwargs = kwargs
        
        if HAS_XGBOOST:
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                eval_metric="logloss",
                **kwargs
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                **kwargs
            )
        
        self.feature_names = None
    
    def fit(self, X, y):
        """
        Train the gradient boosting model.
        
        Args:
            X: Feature matrix (n_samples, n_features) or DataFrame
            y: Target vector (n_samples,) - binary (1 = home win, 0 = away win)
        """
        # Store feature names if DataFrame
        if hasattr(X, "columns"):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        logger.info(f"Training gradient boosting on {len(X)} samples with {X.shape[1]} features")
        logger.info(f"Using {'XGBoost' if HAS_XGBOOST else 'sklearn GradientBoostingClassifier'}")
        
        self.model.fit(X, y)
        
        logger.info("Training complete")
    
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
        
        # Predict probabilities (returns [P(0), P(1)], we want P(1) = home win)
        proba = self.model.predict_proba(X)
        return proba[:, 1]  # Return probability of class 1 (home win)
    
    def get_feature_importance(self):
        """
        Get feature importance.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.feature_names is None:
            return None
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))

