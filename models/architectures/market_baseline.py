"""
Market-Only Baseline Model

A benchmark model that uses ONLY betting market information to predict home win probabilities.
This serves as a baseline to compare against our feature-based models.
"""

import numpy as np
import pandas as pd
from typing import Optional
import logging

from models.base import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def moneyline_to_probability(ml_home: float, ml_away: float) -> float:
    """
    Convert American moneyline odds to implied win probability for home team.
    
    Handles vig (overround) by normalizing probabilities.
    
    Args:
        ml_home: Home team moneyline odds (American format)
        ml_away: Away team moneyline odds (American format)
    
    Returns:
        Implied home win probability (normalized to remove vig)
    """
    def ml_to_decimal(ml):
        """Convert American odds to decimal odds."""
        if ml > 0:
            return (ml + 100) / 100
        else:
            return (abs(ml) + 100) / abs(ml)
    
    # Convert to decimal odds
    dec_home = ml_to_decimal(ml_home)
    dec_away = ml_to_decimal(ml_away)
    
    # Calculate implied probabilities (with vig)
    prob_home_vig = 1 / dec_home
    prob_away_vig = 1 / dec_away
    
    # Normalize to remove vig (so probabilities sum to 1)
    total_prob = prob_home_vig + prob_away_vig
    prob_home = prob_home_vig / total_prob
    
    return prob_home


def spread_to_probability(spread: float) -> float:
    """
    Convert point spread to implied win probability.
    
    Uses a logistic mapping based on historical NFL spread-to-probability relationships.
    
    Formula: p = 1 / (1 + exp(spread / 3))
    
    This approximates that:
    - A 3-point spread ≈ 60% win probability
    - A 7-point spread ≈ 80% win probability
    - A 0-point spread = 50% win probability
    
    Args:
        spread: Point spread from home team perspective (negative = home favored)
    
    Returns:
        Implied home win probability
    """
    # Logistic mapping: p = 1 / (1 + exp(spread / divisor))
    # Divisor of 3 approximates NFL historical spread-to-probability relationship
    divisor = 3.0
    p = 1 / (1 + np.exp(spread / divisor))
    return p


class MarketBaselineModel(BaseModel):
    """
    Market-only baseline model.
    
    Uses only betting market information (moneyline or spread) to predict home win probability.
    No training required - predictions are derived directly from market data.
    """
    
    def __init__(self):
        """Initialize market baseline model."""
        self.feature_names = None
    
    def fit(self, X, y):
        """
        Dummy fit method (no training needed for market baseline).
        
        Args:
            X: Feature matrix (unused)
            y: Target vector (unused)
        """
        logger.info("Market baseline model requires no training - using market data directly")
        pass
    
    def predict_proba(self, X):
        """
        Predict home win probabilities using market data only.
        
        Priority:
        1. Use moneyline if available (most accurate)
        2. Fall back to spread if moneyline not available
        3. Default to 0.5 if no market data
        
        Args:
            X: DataFrame with market columns (close_spread, moneyline_home, moneyline_away)
               or array-like (will try to extract market data)
        
        Returns:
            Array of home win probabilities
        """
        # Handle DataFrame input
        if isinstance(X, pd.DataFrame):
            df = X
        else:
            # If array input, we can't extract market data - return 0.5
            logger.warning("Array input provided but market data requires DataFrame. Returning 0.5.")
            return np.full(len(X), 0.5)
        
        probabilities = []
        
        for idx, row in df.iterrows():
            p_market = None
            
            # Priority 1: Use moneyline if available
            if "moneyline_home" in row.index and "moneyline_away" in row.index:
                ml_home = row.get("moneyline_home")
                ml_away = row.get("moneyline_away")
                
                if pd.notna(ml_home) and pd.notna(ml_away):
                    try:
                        p_market = moneyline_to_probability(float(ml_home), float(ml_away))
                    except (ValueError, TypeError):
                        pass
            
            # Priority 2: Use spread if moneyline not available
            if p_market is None and "close_spread" in row.index:
                spread = row.get("close_spread")
                if pd.notna(spread):
                    try:
                        p_market = spread_to_probability(float(spread))
                    except (ValueError, TypeError):
                        pass
            
            # Priority 3: Default to 0.5 if no market data
            if p_market is None:
                p_market = 0.5
            
            probabilities.append(p_market)
        
        return np.array(probabilities)

