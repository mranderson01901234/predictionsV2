"""
Ensemble Model

Simple ensemble that blends logistic regression and gradient boosting predictions.
"""

import numpy as np
import logging

from models.base import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleModel(BaseModel):
    """
    Ensemble model that combines logistic regression and gradient boosting.
    
    Prediction: p = w * p_gbm + (1 - w) * p_logit
    where w is the ensemble weight (0 = all logit, 1 = all GBM).
    """
    
    def __init__(self, logit_model, gbm_model, weight=0.7):
        """
        Initialize ensemble model.
        
        Args:
            logit_model: Trained logistic regression model
            gbm_model: Trained gradient boosting model
            weight: Ensemble weight for GBM (0-1). Higher = more weight on GBM.
        """
        self.logit_model = logit_model
        self.gbm_model = gbm_model
        self.weight = weight
        
        logger.info(f"Initialized ensemble with weight={weight} (GBM weight)")
    
    def fit(self, X, y):
        """
        Fit ensemble (models should already be trained).
        
        Args:
            X: Feature matrix (unused, models already trained)
            y: Target vector (unused, models already trained)
        """
        logger.info("Ensemble models should be pre-trained. Skipping fit.")
        pass
    
    def predict_proba(self, X):
        """
        Predict home team win probabilities using ensemble.
        
        Args:
            X: Feature matrix (n_samples, n_features) or DataFrame
        
        Returns:
            Array of shape (n_samples,) with blended home win probabilities
        """
        # Get predictions from both models
        p_logit = self.logit_model.predict_proba(X)
        p_gbm = self.gbm_model.predict_proba(X)
        
        # Blend: p = w * p_gbm + (1 - w) * p_logit
        p_ensemble = self.weight * p_gbm + (1 - self.weight) * p_logit
        
        return p_ensemble
    
    def set_weight(self, weight):
        """
        Update ensemble weight.
        
        Args:
            weight: New weight for GBM (0-1)
        """
        if not 0 <= weight <= 1:
            raise ValueError("Weight must be between 0 and 1")
        self.weight = weight
        logger.info(f"Updated ensemble weight to {weight}")

