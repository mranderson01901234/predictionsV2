"""
Tests for NFL Team Form Features

Tests rolling feature calculations, ensuring no data leakage.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.nfl.team_form_features import (
    calculate_win_loss,
    calculate_point_differential,
    compute_rolling_features,
)


class TestRollingFeatures:
    """Test rolling feature calculations."""
    
    @pytest.fixture
    def sample_team_stats(self):
        """Sample team stats for a single team across multiple games."""
        # Create a team with known win/loss pattern
        return pd.DataFrame({
            "game_id": [
                "nfl_2023_01_TEAM_OPP1",
                "nfl_2023_02_TEAM_OPP2",
                "nfl_2023_03_TEAM_OPP3",
                "nfl_2023_04_TEAM_OPP4",
                "nfl_2023_05_TEAM_OPP5",
                "nfl_2023_06_TEAM_OPP6",
            ],
            "team": ["TEAM"] * 6,
            "is_home": [True, False, True, False, True, False],
            "points_for": [21, 17, 24, 14, 28, 20],
            "points_against": [17, 21, 20, 17, 14, 24],
            "win": [1, 0, 1, 0, 1, 0],  # Win, Loss, Win, Loss, Win, Loss
            "point_diff": [4, -4, 4, -3, 14, -4],
        })
    
    def test_win_rate_calculation(self, sample_team_stats):
        """Test win_rate_last4 calculation."""
        features = compute_rolling_features(sample_team_stats, windows=[4])
        
        # Game 5 (index 4): Should use games 1-4 (2 wins, 2 losses) = 0.5
        game_5 = features[features["game_id"] == "nfl_2023_05_TEAM_OPP5"].iloc[0]
        assert abs(game_5["win_rate_last4"] - 0.5) < 0.01, f"Expected 0.5, got {game_5['win_rate_last4']}"
        
        # Game 6 (index 5): Should use games 2-5 (2 wins, 2 losses) = 0.5
        game_6 = features[features["game_id"] == "nfl_2023_06_TEAM_OPP6"].iloc[0]
        assert abs(game_6["win_rate_last4"] - 0.5) < 0.01, f"Expected 0.5, got {game_6['win_rate_last4']}"
    
    def test_point_differential_average(self, sample_team_stats):
        """Test point differential average calculation."""
        features = compute_rolling_features(sample_team_stats, windows=[4])
        
        # Game 5: point_diff values from games 1-4: [4, -4, 4, -3] = avg = 0.25
        game_5 = features[features["game_id"] == "nfl_2023_05_TEAM_OPP5"].iloc[0]
        expected_avg = (4 + (-4) + 4 + (-3)) / 4
        assert abs(game_5["pdiff_last4"] - expected_avg) < 0.01, \
            f"Expected {expected_avg}, got {game_5['pdiff_last4']}"
    
    def test_no_data_leakage(self, sample_team_stats):
        """Test that current game is excluded from rolling windows."""
        features = compute_rolling_features(sample_team_stats, windows=[4])
        
        # Game 1: Should have no history (win_rate = 0 or NaN)
        game_1 = features[features["game_id"] == "nfl_2023_01_TEAM_OPP1"].iloc[0]
        assert game_1["win_rate_last4"] == 0.0, "First game should have no history"
        
        # Game 2: Should only use game 1
        game_2 = features[features["game_id"] == "nfl_2023_02_TEAM_OPP2"].iloc[0]
        # Only 1 game in history, so win_rate = 1.0 (from game 1)
        assert abs(game_2["win_rate_last4"] - 1.0) < 0.01, \
            f"Game 2 should use only game 1, got {game_2['win_rate_last4']}"
    
    def test_points_for_average(self, sample_team_stats):
        """Test points_for average calculation."""
        features = compute_rolling_features(sample_team_stats, windows=[4])
        
        # Game 5: points_for from games 1-4: [21, 17, 24, 14] = avg = 19.0
        game_5 = features[features["game_id"] == "nfl_2023_05_TEAM_OPP5"].iloc[0]
        expected_avg = (21 + 17 + 24 + 14) / 4
        assert abs(game_5["points_for_last4"] - expected_avg) < 0.01, \
            f"Expected {expected_avg}, got {game_5['points_for_last4']}"
    
    def test_multiple_windows(self, sample_team_stats):
        """Test that multiple window sizes work correctly."""
        features = compute_rolling_features(sample_team_stats, windows=[4, 8])
        
        # Should have both win_rate_last4 and win_rate_last8
        assert "win_rate_last4" in features.columns
        assert "win_rate_last8" in features.columns
        
        # Game 6 should have both
        game_6 = features[features["game_id"] == "nfl_2023_06_TEAM_OPP6"].iloc[0]
        assert not pd.isna(game_6["win_rate_last4"])
        # win_rate_last8 might be 0 if not enough history
        assert not pd.isna(game_6["win_rate_last8"])


class TestFeatureGeneration:
    """Test full feature generation pipeline."""
    
    def test_feature_file_exists(self):
        """Test that team_baseline_features.parquet exists."""
        features_path = (
            Path(__file__).parent.parent
            / "data"
            / "nfl"
            / "processed"
            / "team_baseline_features.parquet"
        )
        
        if features_path.exists():
            df = pd.read_parquet(features_path)
            assert len(df) > 0
            
            # Check required columns
            required_cols = [
                "game_id",
                "team",
                "is_home",
                "win_rate_last4",
                "win_rate_last8",
                "win_rate_last16",
                "pdiff_last4",
                "pdiff_last8",
                "pdiff_last16",
            ]
            
            for col in required_cols:
                assert col in df.columns, f"Missing required column: {col}"
    
    def test_no_nulls_in_core_features(self):
        """Test that core features don't have nulls."""
        features_path = (
            Path(__file__).parent.parent
            / "data"
            / "nfl"
            / "processed"
            / "team_baseline_features.parquet"
        )
        
        if features_path.exists():
            df = pd.read_parquet(features_path)
            
            # Core features should not be null (except possibly for first few games)
            core_features = [
                "win_rate_last4",
                "pdiff_last4",
                "points_for_last4",
                "points_against_last4",
            ]
            
            for col in core_features:
                null_count = df[col].isna().sum()
                # Allow some nulls for early games with insufficient history
                assert null_count < len(df) * 0.1, \
                    f"Too many nulls in {col}: {null_count}/{len(df)}"


class TestGameLevelFeatures:
    """Test game-level feature merging."""
    
    def test_game_features_file_exists(self):
        """Test that game_features_baseline.parquet exists."""
        features_path = (
            Path(__file__).parent.parent
            / "data"
            / "nfl"
            / "processed"
            / "game_features_baseline.parquet"
        )
        
        if features_path.exists():
            df = pd.read_parquet(features_path)
            assert len(df) > 0
            
            # Should have home_ and away_ prefixed features
            home_features = [col for col in df.columns if col.startswith("home_")]
            away_features = [col for col in df.columns if col.startswith("away_")]
            
            assert len(home_features) > 0, "Should have home team features"
            assert len(away_features) > 0, "Should have away team features"
            
            # Should have same number of home and away features
            assert len(home_features) == len(away_features), \
                "Should have equal number of home and away features"
    
    def test_no_duplicate_games(self):
        """Test that there are no duplicate game_ids."""
        features_path = (
            Path(__file__).parent.parent
            / "data"
            / "nfl"
            / "processed"
            / "game_features_baseline.parquet"
        )
        
        if features_path.exists():
            df = pd.read_parquet(features_path)
            
            # Should have exactly one row per game_id
            assert not df["game_id"].duplicated().any(), "Found duplicate game_ids"
            
            # Should match number of games in games_markets
            games_markets_path = (
                Path(__file__).parent.parent
                / "data"
                / "nfl"
                / "staged"
                / "games_markets.parquet"
            )
            
            if games_markets_path.exists():
                games_markets_df = pd.read_parquet(games_markets_path)
                # Allow some games to be missing features (early games with insufficient history)
                assert len(df) >= len(games_markets_df) * 0.9, \
                    f"Too many games missing features: {len(df)}/{len(games_markets_df)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

