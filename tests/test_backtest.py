"""
Tests for Backtest ROI Calculation
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.backtest import (
    spread_to_implied_probability,
    compute_market_implied_probabilities,
    calculate_roi,
)


class TestSpreadToProbability:
    """Test spread to probability conversion."""
    
    def test_even_spread(self):
        """Test that 0 spread gives ~0.5 probability."""
        p = spread_to_implied_probability(0.0)
        assert abs(p - 0.5) < 0.1  # Should be close to 0.5
    
    def test_home_favored(self):
        """Test that negative spread (home favored) gives >0.5 probability."""
        p = spread_to_implied_probability(-3.0)
        assert p > 0.5, "Home favored should have >0.5 probability"
    
    def test_away_favored(self):
        """Test that positive spread (away favored) gives <0.5 probability."""
        p = spread_to_implied_probability(3.0)
        assert p < 0.5, "Away favored should have <0.5 probability"


class TestMarketProbabilities:
    """Test market probability calculation."""
    
    def test_spread_based(self):
        """Test calculation from spread."""
        df = pd.DataFrame({
            "close_spread": [-3.0, 0.0, 3.0],
        })
        
        p_market = compute_market_implied_probabilities(df)
        
        assert len(p_market) == 3
        assert p_market.iloc[0] > 0.5  # Home favored
        assert abs(p_market.iloc[1] - 0.5) < 0.1  # Even
        assert p_market.iloc[2] < 0.5  # Away favored
    
    def test_no_market_data(self):
        """Test fallback when no market data."""
        df = pd.DataFrame({
            "other_col": [1, 2, 3],
        })
        
        p_market = compute_market_implied_probabilities(df)
        
        # Should default to 0.5
        assert all(p_market == 0.5)


class TestROICalculation:
    """Test ROI calculation logic."""
    
    def test_no_bets(self):
        """Test ROI when no bets are placed."""
        y_true = np.array([1, 0, 1])
        p_model = np.array([0.5, 0.5, 0.5])
        p_market = np.array([0.5, 0.5, 0.5])
        
        roi_result = calculate_roi(y_true, p_model, p_market, edge_threshold=0.1)
        
        assert roi_result["n_bets"] == 0
        assert roi_result["roi"] == 0.0
    
    def test_all_bets_win(self):
        """Test ROI when all bets win."""
        y_true = np.array([1, 1, 1])
        p_model = np.array([0.8, 0.8, 0.8])  # High model probability
        p_market = np.array([0.5, 0.5, 0.5])  # Market probability
        
        # Edge = 0.8 - 0.5 = 0.3, which is > 0.05 threshold
        roi_result = calculate_roi(y_true, p_model, p_market, edge_threshold=0.05)
        
        assert roi_result["n_bets"] == 3
        assert roi_result["win_rate"] == 1.0
        assert roi_result["roi"] > 0  # Should be positive
    
    def test_all_bets_lose(self):
        """Test ROI when all bets lose."""
        y_true = np.array([0, 0, 0])  # All losses
        p_model = np.array([0.8, 0.8, 0.8])  # High model probability (wrong)
        p_market = np.array([0.5, 0.5, 0.5])
        
        roi_result = calculate_roi(y_true, p_model, p_market, edge_threshold=0.05)
        
        assert roi_result["n_bets"] == 3
        assert roi_result["win_rate"] == 0.0
        assert roi_result["roi"] < 0  # Should be negative
    
    def test_edge_threshold(self):
        """Test that bets only happen when edge >= threshold."""
        y_true = np.array([1, 0, 1, 0])
        p_model = np.array([0.52, 0.53, 0.54, 0.55])  # Small edges: 0.02, 0.03, 0.04, 0.05
        p_market = np.array([0.5, 0.5, 0.5, 0.5])
        
        # With threshold 0.05, only last one should bet (edge >= 0.05)
        roi_result = calculate_roi(y_true, p_model, p_market, edge_threshold=0.05)
        
        assert roi_result["n_bets"] == 1  # Only last game (edge exactly 0.05)
    
    def test_mixed_outcomes(self):
        """Test ROI with mixed win/loss outcomes."""
        y_true = np.array([1, 0, 1, 0, 1])
        p_model = np.array([0.7, 0.7, 0.7, 0.7, 0.7])
        p_market = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        
        roi_result = calculate_roi(y_true, p_model, p_market, edge_threshold=0.05)
        
        assert roi_result["n_bets"] == 5
        assert roi_result["win_rate"] == 0.6  # 3 wins / 5 bets
        # ROI should be positive if win rate > 0.5 (simplified unit bet)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

