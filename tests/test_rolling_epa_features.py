"""
Tests for NFL Rolling EPA Features

Validates rolling window calculations, no data leakage, and early-season handling.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.nfl.rolling_epa_features import compute_rolling_epa_features


class TestRollingEPAFeatures:
    """Test rolling EPA feature calculations."""
    
    @pytest.fixture
    def sample_team_epa_features(self):
        """Create synthetic team EPA features for testing."""
        # Create 10 games for a team to test rolling windows
        return pd.DataFrame({
            "game_id": [
                f"nfl_2023_{i:02d}_TEAM_OPP{i}" for i in range(1, 11)
            ],
            "team": ["TEAM"] * 10,
            "offensive_epa_per_play": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1],
            "offensive_pass_epa": [0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.2],
            "offensive_run_epa": [0.0, 0.1, 0.2, 0.0, 0.1, 0.2, 0.0, 0.1, 0.2, 0.0],
            "offensive_success_rate": [0.5, 0.6, 0.7, 0.5, 0.6, 0.7, 0.5, 0.6, 0.7, 0.5],
            "offensive_pass_success_rate": [0.6, 0.7, 0.8, 0.6, 0.7, 0.8, 0.6, 0.7, 0.8, 0.6],
            "offensive_run_success_rate": [0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.4],
            "defensive_epa_per_play_allowed": [-0.1, -0.2, -0.3, -0.1, -0.2, -0.3, -0.1, -0.2, -0.3, -0.1],
            "defensive_pass_epa_allowed": [-0.2, -0.3, -0.4, -0.2, -0.3, -0.4, -0.2, -0.3, -0.4, -0.2],
            "defensive_run_epa_allowed": [0.0, -0.1, -0.2, 0.0, -0.1, -0.2, 0.0, -0.1, -0.2, 0.0],
            "defensive_success_rate_allowed": [0.5, 0.4, 0.3, 0.5, 0.4, 0.3, 0.5, 0.4, 0.3, 0.5],
            "defensive_pass_success_rate_allowed": [0.4, 0.3, 0.2, 0.4, 0.3, 0.2, 0.4, 0.3, 0.2, 0.4],
            "defensive_run_success_rate_allowed": [0.6, 0.5, 0.4, 0.6, 0.5, 0.4, 0.6, 0.5, 0.4, 0.6],
        })
    
    @pytest.fixture
    def sample_games(self):
        """Create sample games DataFrame for date ordering."""
        return pd.DataFrame({
            "game_id": [f"nfl_2023_{i:02d}_TEAM_OPP{i}" for i in range(1, 11)],
            "date": pd.to_datetime([f"2023-09-{i:02d}" for i in range(10, 20)]),
            "season": [2023] * 10,
            "week": list(range(1, 11)),
        })
    
    def test_rolling_windows_exclude_current_game(self, sample_team_epa_features, sample_games):
        """Test that rolling windows exclude the current game."""
        rolling_df = compute_rolling_epa_features(
            sample_team_epa_features, sample_games, windows=[3]
        )
        
        # Game 4: Should use games 1-3 (exclude game 4)
        game_4 = rolling_df[rolling_df["game_id"] == "nfl_2023_04_TEAM_OPP4"].iloc[0]
        expected_epa = (0.1 + 0.2 + 0.3) / 3  # Games 1, 2, 3
        assert abs(game_4["off_epa_last3"] - expected_epa) < 0.01
        
        # Game 5: Should use games 2-4 (exclude game 5)
        game_5 = rolling_df[rolling_df["game_id"] == "nfl_2023_05_TEAM_OPP5"].iloc[0]
        expected_epa = (0.2 + 0.3 + 0.1) / 3  # Games 2, 3, 4
        assert abs(game_5["off_epa_last3"] - expected_epa) < 0.01
    
    def test_multiple_window_sizes(self, sample_team_epa_features, sample_games):
        """Test that multiple window sizes work correctly."""
        rolling_df = compute_rolling_epa_features(
            sample_team_epa_features, sample_games, windows=[3, 5, 8]
        )
        
        # Game 6: Should have last3, last5, last8
        game_6 = rolling_df[rolling_df["game_id"] == "nfl_2023_06_TEAM_OPP6"].iloc[0]
        
        # last3: games 3, 4, 5
        expected_last3 = (0.3 + 0.1 + 0.2) / 3
        assert abs(game_6["off_epa_last3"] - expected_last3) < 0.01
        
        # last5: games 1, 2, 3, 4, 5
        expected_last5 = (0.1 + 0.2 + 0.3 + 0.1 + 0.2) / 5
        assert abs(game_6["off_epa_last5"] - expected_last5) < 0.01
        
        # last8: games 1-5 (only 5 available before game 6)
        expected_last8 = (0.1 + 0.2 + 0.3 + 0.1 + 0.2) / 5  # Uses available games
        assert abs(game_6["off_epa_last8"] - expected_last8) < 0.01
    
    def test_early_season_handling(self, sample_team_epa_features, sample_games):
        """Test handling of early-season games with insufficient history."""
        rolling_df = compute_rolling_epa_features(
            sample_team_epa_features, sample_games, windows=[3, 5, 8]
        )
        
        # Game 1: No history, should be None/NaN
        game_1 = rolling_df[rolling_df["game_id"] == "nfl_2023_01_TEAM_OPP1"].iloc[0]
        assert pd.isna(game_1["off_epa_last3"]) or game_1["off_epa_last3"] is None
        
        # Game 2: Only 1 game in history, should use that
        game_2 = rolling_df[rolling_df["game_id"] == "nfl_2023_02_TEAM_OPP2"].iloc[0]
        # Should use game 1 only (partial window)
        assert abs(game_2["off_epa_last3"] - 0.1) < 0.01  # Only game 1
        
        # Game 3: 2 games in history
        game_3 = rolling_df[rolling_df["game_id"] == "nfl_2023_03_TEAM_OPP3"].iloc[0]
        expected = (0.1 + 0.2) / 2  # Games 1, 2
        assert abs(game_3["off_epa_last3"] - expected) < 0.01
    
    def test_defensive_metrics(self, sample_team_epa_features, sample_games):
        """Test that defensive metrics are computed correctly."""
        rolling_df = compute_rolling_epa_features(
            sample_team_epa_features, sample_games, windows=[3]
        )
        
        # Game 4: Defensive EPA allowed (last 3 games)
        game_4 = rolling_df[rolling_df["game_id"] == "nfl_2023_04_TEAM_OPP4"].iloc[0]
        expected_def_epa = (-0.1 + -0.2 + -0.3) / 3
        assert abs(game_4["def_epa_allowed_last3"] - expected_def_epa) < 0.01
    
    def test_pass_run_splits(self, sample_team_epa_features, sample_games):
        """Test that pass/run splits are computed correctly."""
        rolling_df = compute_rolling_epa_features(
            sample_team_epa_features, sample_games, windows=[3]
        )
        
        # Game 4: Pass EPA (last 3 games)
        game_4 = rolling_df[rolling_df["game_id"] == "nfl_2023_04_TEAM_OPP4"].iloc[0]
        expected_pass_epa = (0.2 + 0.3 + 0.4) / 3
        assert abs(game_4["off_pass_epa_last3"] - expected_pass_epa) < 0.01
        
        # Run EPA
        expected_run_epa = (0.0 + 0.1 + 0.2) / 3
        assert abs(game_4["off_run_epa_last3"] - expected_run_epa) < 0.01
    
    def test_success_rate_metrics(self, sample_team_epa_features, sample_games):
        """Test that success rate metrics are computed correctly."""
        rolling_df = compute_rolling_epa_features(
            sample_team_epa_features, sample_games, windows=[3]
        )
        
        # Game 4: Success rate (last 3 games)
        game_4 = rolling_df[rolling_df["game_id"] == "nfl_2023_04_TEAM_OPP4"].iloc[0]
        expected_sr = (0.5 + 0.6 + 0.7) / 3
        assert abs(game_4["off_sr_last3"] - expected_sr) < 0.01
    
    def test_no_data_leakage(self, sample_team_epa_features, sample_games):
        """Test that there is no data leakage (current game excluded)."""
        rolling_df = compute_rolling_epa_features(
            sample_team_epa_features, sample_games, windows=[3]
        )
        
        # Game 4 EPA is 0.1
        # Rolling last3 for game 4 should NOT include game 4 itself
        game_4 = rolling_df[rolling_df["game_id"] == "nfl_2023_04_TEAM_OPP4"].iloc[0]
        # Should be average of games 1, 2, 3: (0.1 + 0.2 + 0.3) / 3 = 0.2
        assert abs(game_4["off_epa_last3"] - 0.2) < 0.01
        
        # If game 4 were included, it would be (0.1 + 0.2 + 0.3 + 0.1) / 4 = 0.175
        # So we verify it's NOT 0.175
        assert abs(game_4["off_epa_last3"] - 0.175) > 0.01

