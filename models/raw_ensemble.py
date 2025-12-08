"""
Raw Ensemble Wrapper

Bypasses calibration to use raw (uncalibrated) predictions from base models.
This fixes the calibration issue where isotonic regression is overfitting.

Usage:
    from models.raw_ensemble import RawEnsemble
    
    # Load original model
    model = StackingEnsemble.load(path)
    
    # Wrap to use raw predictions
    raw_model = RawEnsemble(model)
    
    # Use like normal model
    probs = raw_model.predict_proba(X)[:, 1]
"""

import numpy as np
from typing import Optional
from models.base import BaseModel


class RawEnsemble(BaseModel):
    """
    Wrapper to use raw (uncalibrated) predictions from stacking ensemble.
    
    This bypasses any calibration layer that may be hurting performance.
    """
    
    def __init__(self, model: BaseModel):
        """
        Initialize wrapper.
        
        Args:
            model: Original stacking ensemble model
        """
        self.model = model
        
        # Verify model has base_models
        if not hasattr(model, 'base_models'):
            raise ValueError(
                "Model must have 'base_models' attribute. "
                "This wrapper is for StackingEnsemble models."
            )
    
    def predict_proba(self, X) -> np.ndarray:
        """
        Get raw (uncalibrated) predictions by averaging base model predictions.
        
        Args:
            X: Feature matrix
        
        Returns:
            Array of shape (n_samples, 2) with [away_prob, home_prob]
        """
        # Average predictions from base models
        preds = []
        
        for name, base_model in self.model.base_models.items():
            if hasattr(base_model, 'predict_proba'):
                prob = base_model.predict_proba(X)
                
                # Handle 2D array (binary classification)
                if prob.ndim == 2:
                    if prob.shape[1] == 2:
                        prob = prob[:, 1]  # Home win probability
                    else:
                        prob = prob[:, 0]
                
                preds.append(prob)
        
        if not preds:
            raise ValueError("No base models with predict_proba method found")
        
        # Average probabilities
        avg_prob = np.mean(preds, axis=0)
        
        # Return in sklearn format: [away_prob, home_prob]
        return np.column_stack([1 - avg_prob, avg_prob])
    
    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary outcomes.
        
        Args:
            X: Feature matrix
            threshold: Probability threshold for classification
        
        Returns:
            Binary predictions (1 = home win, 0 = away win)
        """
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)
    
    def fit(self, X, y):
        """
        No-op: model is already trained.
        
        This is required by BaseModel interface but does nothing
        since we're wrapping an already-trained model.
        """
        pass
    
    def save(self, path):
        """
        Save wrapper (saves reference to original model).
        
        Note: Original model must still be available at load time.
        """
        import pickle
        from pathlib import Path
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model path reference (if model has a path)
        # Otherwise, save the model itself
        save_dict = {
            'model': self.model,
            'type': 'RawEnsemble',
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
    
    @classmethod
    def load(cls, path):
        """Load wrapper."""
        import pickle
        from pathlib import Path
        
        path = Path(path)
        
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        return cls(save_dict['model'])

