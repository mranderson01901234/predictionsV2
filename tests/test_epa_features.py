"""
Tests for NFL EPA Features

Tests EPA metric calculations using synthetic play-by-play data.
Validates EPA averages, success rates, offensive vs defensive inversion,
and situational splits.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.nfl.epa_features import compute_team_epa_features


class TestEPAFeatures:
    """Test EPA feature calculations."""
    
    @pytest.fixture
    def synthetic_plays(self):
        """Create synthetic play-by-play data for testing."""
        # All plays are from the same game: TEAM (away) @ OPP (home)
        # So game_id should be consistent: nfl_2023_01_TEAM_OPP
        return pd.DataFrame({
            "game_id": [
                "nfl_2023_01_TEAM_OPP",
                "nfl_2023_01_TEAM_OPP",
                "nfl_2023_01_TEAM_OPP",
                "nfl_2023_01_TEAM_OPP",
                "nfl_2023_01_TEAM_OPP",
                "nfl_2023_01_TEAM_OPP",
                "nfl_2023_01_TEAM_OPP",  # OPP on offense (same game)
                "nfl_2023_01_TEAM_OPP",
                "nfl_2023_01_TEAM_OPP",
            ],
            "play_id": [1, 2, 3, 4, 5, 6, 1, 2, 3],
            "posteam": ["TEAM", "TEAM", "TEAM", "TEAM", "TEAM", "TEAM", "OPP", "OPP", "OPP"],
            "defteam": ["OPP", "OPP", "OPP", "OPP", "OPP", "OPP", "TEAM", "TEAM", "TEAM"],
            "play_type": ["pass", "run", "pass", "run", "pass", "pass", "pass", "run", "pass"],
            "epa": [0.5, -0.2, 0.3, 0.1, 0.4, 0.2, -0.3, -0.1, -0.2],  # TEAM: [0.5, -0.2, 0.3, 0.1, 0.4, 0.2] = avg 0.22
            "success": [1, 0, 1, 1, 1, 1, 0, 0, 0],  # TEAM: 5/6 = 0.833
            "is_pass": [1, 0, 1, 0, 1, 1, 1, 0, 1],
            "is_run": [0, 1, 0, 1, 0, 0, 0, 1, 0],
            "down": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "ydstogo": [10, 5, 3, 10, 7, 3, 10, 5, 8],
            "yardline_100": [50, 45, 15, 60, 55, 18, 50, 45, 60],  # One red zone play (15, 18)
            "qtr": [1, 1, 2, 1, 1, 2, 1, 1, 2],
            "half_seconds_remaining": [1800, 1500, 1200, 1800, 1600, 1100, 1800, 1500, 1200],
        })
    
    def test_epa_averages(self, synthetic_plays):
        """Test that EPA averages match expected values."""
        features_df = compute_team_epa_features(synthetic_plays)
        
        # TEAM offensive EPA - filter by game_id and team
        team_features = features_df[
            (features_df["team"] == "TEAM") & 
            (features_df["game_id"] == "nfl_2023_01_TEAM_OPP")
        ].iloc[0]
        
        # Expected: (0.5 + -0.2 + 0.3 + 0.1 + 0.4 + 0.2) / 6 = 0.2167
        expected_epa = (0.5 + -0.2 + 0.3 + 0.1 + 0.4 + 0.2) / 6
        assert abs(team_features["offensive_epa_per_play"] - expected_epa) < 0.01
        
        # Pass EPA: (0.5 + 0.3 + 0.4 + 0.2) / 4 = 0.35
        expected_pass_epa = (0.5 + 0.3 + 0.4 + 0.2) / 4
        assert abs(team_features["offensive_pass_epa"] - expected_pass_epa) < 0.01
        
        # Run EPA: (-0.2 + 0.1) / 2 = -0.05
        expected_run_epa = (-0.2 + 0.1) / 2
        assert abs(team_features["offensive_run_epa"] - expected_run_epa) < 0.01
    
    def test_success_rate(self, synthetic_plays):
        """Test that success rate matches expected values."""
        features_df = compute_team_epa_features(synthetic_plays)
        
        team_features = features_df[
            (features_df["team"] == "TEAM") & 
            (features_df["game_id"] == "nfl_2023_01_TEAM_OPP")
        ].iloc[0]
        
        # Expected success rate: 5/6 = 0.8333
        expected_sr = 5 / 6
        assert abs(team_features["offensive_success_rate"] - expected_sr) < 0.01
        
        # Pass success rate: 4/4 = 1.0
        expected_pass_sr = 4 / 4
        assert abs(team_features["offensive_pass_success_rate"] - expected_pass_sr) < 0.01
        
        # Run success rate: 1/2 = 0.5
        expected_run_sr = 1 / 2
        assert abs(team_features["offensive_run_success_rate"] - expected_run_sr) < 0.01
    
    def test_defensive_inversion(self, synthetic_plays):
        """Test that defensive metrics invert correctly."""
        features_df = compute_team_epa_features(synthetic_plays)
        
        team_features = features_df[
            (features_df["team"] == "TEAM") & 
            (features_df["game_id"] == "nfl_2023_01_TEAM_OPP")
        ].iloc[0]
        
        # TEAM defense vs OPP offense
        # OPP EPA: (-0.3 + -0.1 + -0.2) / 3 = -0.2
        # TEAM defensive EPA allowed = -(-0.2) = 0.2 (good defense allows negative EPA)
        opp_epa = (-0.3 + -0.1 + -0.2) / 3
        expected_def_epa_allowed = -opp_epa  # Negate because defense perspective
        
        assert abs(team_features["defensive_epa_per_play_allowed"] - expected_def_epa_allowed) < 0.01
        
        # OPP success: 0/3 = 0.0
        # TEAM defensive success rate allowed = 1 - 0.0 = 1.0 (good defense prevents success)
        opp_success = 0 / 3
        expected_def_sr_allowed = 1.0 - opp_success
        assert abs(team_features["defensive_success_rate_allowed"] - expected_def_sr_allowed) < 0.01
    
    def test_situational_splits(self, synthetic_plays):
        """Test that situational splits match filters."""
        features_df = compute_team_epa_features(synthetic_plays)
        
        team_features = features_df[
            (features_df["team"] == "TEAM") & 
            (features_df["game_id"] == "nfl_2023_01_TEAM_OPP")
        ].iloc[0]
        
        # Third down EPA: (0.3 + 0.2) / 2 = 0.25
        expected_third_down = (0.3 + 0.2) / 2
        assert abs(team_features["third_down_epa"] - expected_third_down) < 0.01
        
        # Red zone EPA (yardline_100 <= 20): (0.3 + 0.2) / 2 = 0.25
        # Plays at yardline 15 and 18
        expected_red_zone = (0.3 + 0.2) / 2
        assert abs(team_features["red_zone_epa"] - expected_red_zone) < 0.01
        
        # Late downs (3rd/4th): (0.3 + 0.2) / 2 = 0.25
        expected_late_down = (0.3 + 0.2) / 2
        assert abs(team_features["late_down_epa"] - expected_late_down) < 0.01
        
        # Early downs (1st/2nd): (0.5 + -0.2 + 0.1 + 0.4) / 4 = 0.2
        expected_early_down = (0.5 + -0.2 + 0.1 + 0.4) / 4
        assert abs(team_features["early_down_epa"] - expected_early_down) < 0.01
    
    def test_no_data_leakage(self, synthetic_plays):
        """Test that EPA is computed per-game only (no rolling)."""
        features_df = compute_team_epa_features(synthetic_plays)
        
        # Each game_id should have exactly 2 rows (one per team)
        game_counts = features_df.groupby("game_id").size()
        assert all(game_counts == 2), "Each game should have exactly 2 team rows"
        
        # Each team-game combination should be unique
        assert not features_df.duplicated(subset=["game_id", "team"]).any()
    
    def test_epa_ranges(self, synthetic_plays):
        """Test that EPA values are in reasonable ranges."""
        features_df = compute_team_epa_features(synthetic_plays)
        
        # Offensive EPA should be between -1.0 and 1.0 (typical range: -0.2 to 0.3)
        assert all(features_df["offensive_epa_per_play"] >= -1.0)
        assert all(features_df["offensive_epa_per_play"] <= 1.0)
        
        # Success rates should be between 0 and 1
        assert all(features_df["offensive_success_rate"] >= 0.0)
        assert all(features_df["offensive_success_rate"] <= 1.0)
        
        # Defensive EPA allowed should also be in reasonable range
        assert all(features_df["defensive_epa_per_play_allowed"] >= -1.0)
        assert all(features_df["defensive_epa_per_play_allowed"] <= 1.0)
    
    def test_empty_play_types(self):
        """Test handling of plays with no pass/run plays."""
        # Create plays with only special teams plays
        # These should be filtered out, so we expect no rows for this team-game
        empty_plays = pd.DataFrame({
            "game_id": ["nfl_2023_01_TEAM_OPP"],
            "play_id": [1],
            "posteam": ["TEAM"],
            "defteam": ["OPP"],
            "play_type": ["field_goal"],
            "epa": [0.5],
            "success": [1],
            "is_pass": [0],
            "is_run": [0],
            "down": [4],
            "ydstogo": [0],
            "yardline_100": [30],
            "qtr": [2],
            "half_seconds_remaining": [1200],
        })
        
        # Should raise ValueError because no valid offensive plays (no pass/run)
        with pytest.raises(ValueError, match="No valid offensive plays"):
            compute_team_epa_features(empty_plays)

