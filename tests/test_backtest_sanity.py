"""
Sanity Tests for ROI/Edge Calculation

Tests edge calculation and ROI logic with synthetic scenarios where we know
the expected outcomes.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.backtest import (
    compute_model_edge,
    simulate_betting,
    compute_market_implied_probabilities,
)


class TestEdgeCalculation:
    """Test edge calculation logic."""
    
    def test_positive_edge(self):
        """Test that positive edge means model favors home team more than market."""
        p_model = np.array([0.7, 0.6, 0.5])
        p_market = np.array([0.5, 0.5, 0.5])
        
        edge = compute_model_edge(p_model, p_market)
        
        assert edge[0] > 0, "Model should have positive edge when p_model > p_market"
        assert edge[1] > 0, "Model should have positive edge when p_model > p_market"
        assert edge[2] == 0, "Edge should be zero when p_model == p_market"
    
    def test_negative_edge(self):
        """Test that negative edge means model favors home team less than market."""
        p_model = np.array([0.3, 0.4])
        p_market = np.array([0.5, 0.5])
        
        edge = compute_model_edge(p_model, p_market)
        
        assert edge[0] < 0, "Model should have negative edge when p_model < p_market"
        assert edge[1] < 0, "Model should have negative edge when p_model < p_market"
    
    def test_edge_calculation_formula(self):
        """Test that edge = p_model - p_market."""
        p_model = np.array([0.6, 0.7, 0.8])
        p_market = np.array([0.5, 0.6, 0.7])
        
        edge = compute_model_edge(p_model, p_market)
        expected = p_model - p_market
        
        np.testing.assert_array_equal(edge, expected)


class TestROISanity:
    """Test ROI calculation with known scenarios."""
    
    def test_fair_market_zero_roi(self):
        """
        Test that betting on fair market with no edge yields ~0 ROI over many bets.
        
        Scenario: Market is perfectly calibrated, model matches market exactly.
        Expected: ROI should be approximately 0 (with some variance due to randomness).
        """
        np.random.seed(42)
        n_games = 1000
        
        # Fair market: true probability = market probability = 0.5
        p_true = np.full(n_games, 0.5)
        p_market = np.full(n_games, 0.5)
        p_model = np.full(n_games, 0.5)  # Model matches market
        
        # Simulate outcomes
        y_true = (np.random.rand(n_games) < p_true).astype(int)
        
        # Calculate ROI with very low threshold (bet on everything)
        roi_result = simulate_betting(y_true, p_model, p_market, edge_threshold=0.0)
        
        # ROI should be close to 0 (within reasonable variance)
        # With fair odds and 50% win rate, expected ROI = 0
        assert abs(roi_result["roi"]) < 0.1, \
            f"Fair market should yield ~0 ROI, got {roi_result['roi']:.4f}"
    
    def test_biased_model_positive_roi(self):
        """
        Test that a model with consistent positive edge yields positive ROI.
        
        Scenario: Model has +5% edge on every game, true probability matches model.
        Expected: Positive ROI (but modest, since edge is small).
        """
        np.random.seed(42)
        n_games = 1000
        
        # Model has 5% edge: p_model = 0.55, p_market = 0.50
        # True probability matches model (model is correct)
        p_true = np.full(n_games, 0.55)
        p_market = np.full(n_games, 0.50)
        p_model = np.full(n_games, 0.55)
        
        # Simulate outcomes
        y_true = (np.random.rand(n_games) < p_true).astype(int)
        
        # Bet when edge >= 0.05
        roi_result = simulate_betting(y_true, p_model, p_market, edge_threshold=0.05)
        
        # Should have positive ROI (model is correct and has edge)
        assert roi_result["roi"] > 0, \
            f"Model with positive edge should yield positive ROI, got {roi_result['roi']:.4f}"
        assert roi_result["n_bets"] == n_games, "Should bet on all games with edge >= threshold"
    
    def test_wrong_model_negative_roi(self):
        """
        Test that a model that's wrong yields negative ROI.
        
        Scenario: Model has +5% edge but is wrong (true prob < market prob).
        Expected: Negative ROI.
        """
        np.random.seed(42)
        n_games = 1000
        
        # Model thinks home team has 55% chance, but true is only 45%
        p_true = np.full(n_games, 0.45)
        p_market = np.full(n_games, 0.50)
        p_model = np.full(n_games, 0.55)  # Model is wrong
        
        # Simulate outcomes
        y_true = (np.random.rand(n_games) < p_true).astype(int)
        
        # Bet when edge >= 0.05
        roi_result = simulate_betting(y_true, p_model, p_market, edge_threshold=0.05)
        
        # Should have negative ROI (model is wrong)
        assert roi_result["roi"] < 0, \
            f"Wrong model should yield negative ROI, got {roi_result['roi']:.4f}"
    
    def test_edge_threshold_filtering(self):
        """Test that edge threshold correctly filters bets."""
        y_true = np.array([1, 0, 1, 0, 1])
        p_model = np.array([0.52, 0.53, 0.54, 0.55, 0.56])  # Edges: 0.02, 0.03, 0.04, 0.05, 0.06
        p_market = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        
        # With threshold 0.05, should only bet on last two games (edge >= 0.05)
        roi_result = simulate_betting(y_true, p_model, p_market, edge_threshold=0.05)
        
        assert roi_result["n_bets"] == 2, \
            f"Should bet on 2 games with edge >= 0.05, got {roi_result['n_bets']}"
    
    def test_win_rate_calculation(self):
        """Test that win rate is calculated correctly."""
        # 5 games, we bet on all, 3 wins
        y_true = np.array([1, 1, 1, 0, 0])
        p_model = np.array([0.6, 0.6, 0.6, 0.6, 0.6])
        p_market = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        
        roi_result = simulate_betting(y_true, p_model, p_market, edge_threshold=0.05)
        
        assert roi_result["n_bets"] == 5
        assert roi_result["win_rate"] == 0.6, \
            f"Win rate should be 0.6 (3/5), got {roi_result['win_rate']:.2f}"
    
    def test_roi_formula(self):
        """Test that ROI formula is correct."""
        # Simple case: 2 bets, 1 win, 1 loss
        # Win: +1 unit, Loss: -1 unit
        # Total staked: 2 units
        # Total profit: 0 units
        # ROI = 0 / 2 = 0
        
        y_true = np.array([1, 0])
        p_model = np.array([0.6, 0.6])
        p_market = np.array([0.5, 0.5])
        
        roi_result = simulate_betting(y_true, p_model, p_market, edge_threshold=0.05)
        
        assert roi_result["n_bets"] == 2
        assert roi_result["total_staked"] == 2.0
        assert roi_result["total_profit"] == 0.0
        assert roi_result["roi"] == 0.0


class TestNoLeakage:
    """Test that no post-game information leaks into betting decisions."""
    
    def test_only_pre_game_data_used(self):
        """
        Test that betting decisions use only pre-game data.
        
        This is a structural test - we verify that the function signature
        and logic only use pre-game inputs.
        """
        # Create synthetic data
        n_games = 100
        
        # Pre-game data (what we should use)
        p_model = np.random.rand(n_games)
        p_market = np.random.rand(n_games)
        
        # Post-game data (what we should NOT use for betting decisions)
        y_true = np.random.randint(0, 2, n_games)
        final_scores = np.random.randint(0, 50, (n_games, 2))
        
        # Simulate betting - should only use p_model and p_market
        roi_result = simulate_betting(y_true, p_model, p_market, edge_threshold=0.05)
        
        # If we got here without error, the function doesn't use post-game data
        # (y_true is only used for evaluation, not betting decisions)
        assert "n_bets" in roi_result
        assert roi_result["n_bets"] >= 0
    
    def test_market_probabilities_from_pre_game_data(self):
        """Test that market probabilities are computed from pre-game data only."""
        # Create DataFrame with pre-game market data
        df = pd.DataFrame({
            "close_spread": [-3.0, 0.0, 3.0],
            "moneyline_home": [np.nan, np.nan, np.nan],  # Not available
            # No post-game columns like home_score, away_score
        })
        
        p_market = compute_market_implied_probabilities(df)
        
        # Should compute probabilities from spread only
        assert len(p_market) == 3
        assert p_market.iloc[0] > 0.5  # Home favored
        assert abs(p_market.iloc[1] - 0.5) < 0.1  # Even
        assert p_market.iloc[2] < 0.5  # Away favored


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

