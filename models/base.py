"""
Base Model Interface

Defines the interface that all prediction models must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional
import pickle
from pathlib import Path


class BaseModel(ABC):
    """
    Base class for all prediction models.
    
    All models must implement:
    - fit(X, y): Train the model
    - predict_proba(X): Return win probabilities for home team
    - save(path): Save model to disk
    - load(path): Load model from disk
    """
    
    @abstractmethod
    def fit(self, X, y):
        """
        Train the model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,) - binary (1 = home win, 0 = away win)
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        """
        Predict home team win probabilities.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Array of shape (n_samples,) with home win probabilities
        """
        pass
    
    def predict(self, X, threshold=0.5):
        """
        Predict binary outcomes (home win = 1, away win = 0).
        
        Args:
            X: Feature matrix
            threshold: Probability threshold for classification
        
        Returns:
            Binary predictions
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def save(self, path: Path):
        """
        Save model to disk.
        
        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: Path):
        """
        Load model from disk.
        
        Args:
            path: Path to model file
        
        Returns:
            Loaded model instance
        """
        path = Path(path)
        
        with open(path, "rb") as f:
            return pickle.load(f)

