"""
Tests for NFL QB Features

Validates QB identification, metrics calculations, and edge cases.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.nfl.qb_features import (
    identify_starting_qb,
    compute_qb_metrics,
    compute_team_qb_features,
)


class TestQBFeatures:
    """Test QB feature calculations."""
    
    @pytest.fixture
    def sample_plays_with_qb(self):
        """Create synthetic play-by-play data with QB information."""
        return pd.DataFrame({
            "game_id": [
                "nfl_2023_01_TEAM_OPP",
                "nfl_2023_01_TEAM_OPP",
                "nfl_2023_01_TEAM_OPP",
                "nfl_2023_01_TEAM_OPP",
                "nfl_2023_01_TEAM_OPP",
                "nfl_2023_01_TEAM_OPP",
                "nfl_2023_01_TEAM_OPP",  # OPP on offense
                "nfl_2023_01_TEAM_OPP",
            ],
            "play_id": [1, 2, 3, 4, 5, 6, 1, 2],
            "posteam": ["TEAM", "TEAM", "TEAM", "TEAM", "TEAM", "TEAM", "OPP", "OPP"],
            "defteam": ["OPP", "OPP", "OPP", "OPP", "OPP", "OPP", "TEAM", "TEAM"],
            "is_pass": [1, 1, 1, 1, 1, 0, 1, 1],  # 5 pass plays for TEAM, 2 for OPP
            "passer_id": ["QB1", "QB1", "QB1", "QB1", "QB2", None, "QB3", "QB3"],  # QB1 has 4, QB2 has 1
            "epa": [0.5, 0.3, 0.4, 0.2, 0.1, 0.0, -0.2, -0.1],
            "success": [1, 1, 1, 0, 1, 0, 0, 0],
            "sack": [0, 0, 1, 0, 0, 0, 0, 0],  # 1 sack for QB1
            "interception": [0, 0, 0, 1, 0, 0, 0, 0],  # 1 INT for QB1
            "air_yards": [10, 8, 5, 12, 6, 0, 7, 9],
        })
    
    def test_starting_qb_identification(self, sample_plays_with_qb):
        """Test that starting QB is identified correctly (most dropbacks)."""
        # QB1 has 4 pass attempts + 1 sack = 5 dropbacks
        # QB2 has 1 pass attempt = 1 dropback
        # QB1 should be selected
        qb_id = identify_starting_qb(sample_plays_with_qb, "nfl_2023_01_TEAM_OPP", "TEAM")
        assert qb_id == "QB1"
    
    def test_qb_metrics_calculation(self, sample_plays_with_qb):
        """Test QB metrics calculations."""
        # QB1 metrics:
        # - Dropbacks: 4 passes + 1 sack = 5
        # - Total EPA: 0.5 + 0.3 + 0.4 + 0.2 = 1.4
        # - EPA per dropback: 1.4 / 5 = 0.28
        # - Success rate: 3/4 = 0.75 (sack doesn't count for success)
        # - Sack rate: 1/5 = 0.2
        # - INT rate: 1/4 = 0.25
        
        metrics = compute_qb_metrics(sample_plays_with_qb, "nfl_2023_01_TEAM_OPP", "TEAM", "QB1")
        
        assert abs(metrics["qb_epa_per_dropback"] - 0.28) < 0.01
        assert abs(metrics["qb_sack_rate"] - 0.2) < 0.01
        assert abs(metrics["qb_int_rate"] - 0.25) < 0.01
        assert metrics["qb_dropbacks"] == 5
    
    def test_qb_features_per_team_game(self, sample_plays_with_qb):
        """Test that QB features are computed per team per game."""
        qb_features_df = compute_team_qb_features(sample_plays_with_qb)
        
        # Should have 2 rows (TEAM and OPP)
        assert len(qb_features_df) == 2
        
        # TEAM should have QB1
        team_qb = qb_features_df[qb_features_df["team"] == "TEAM"].iloc[0]
        assert team_qb["qb_id"] == "QB1"
        
        # OPP should have QB3
        opp_qb = qb_features_df[qb_features_df["team"] == "OPP"].iloc[0]
        assert opp_qb["qb_id"] == "QB3"
    
    def test_no_qb_found(self):
        """Test handling when no QB is found."""
        plays_no_qb = pd.DataFrame({
            "game_id": ["nfl_2023_01_TEAM_OPP"],
            "play_id": [1],
            "posteam": ["TEAM"],
            "defteam": ["OPP"],
            "is_pass": [0],  # No pass plays
            "passer_id": [None],
            "epa": [0.0],
            "success": [0],
        })
        
        qb_features_df = compute_team_qb_features(plays_no_qb)
        
        # Should still create a row, but with null QB metrics
        team_qb = qb_features_df[qb_features_df["team"] == "TEAM"].iloc[0]
        assert team_qb["qb_id"] is None or pd.isna(team_qb["qb_id"])
        assert team_qb["qb_dropbacks"] == 0
    
    def test_sack_rate_range(self, sample_plays_with_qb):
        """Test that sack rate is between 0 and 1."""
        qb_features_df = compute_team_qb_features(sample_plays_with_qb)
        
        # All sack rates should be between 0 and 1
        sack_rates = qb_features_df["qb_sack_rate"].dropna()
        assert all(sack_rates >= 0)
        assert all(sack_rates <= 1)
    
    def test_int_rate_range(self, sample_plays_with_qb):
        """Test that INT rate is between 0 and 1."""
        qb_features_df = compute_team_qb_features(sample_plays_with_qb)
        
        # All INT rates should be between 0 and 1
        int_rates = qb_features_df["qb_int_rate"].dropna()
        assert all(int_rates >= 0)
        assert all(int_rates <= 1)
    
    def test_epa_ranges(self, sample_plays_with_qb):
        """Test that EPA per dropback is in reasonable range."""
        qb_features_df = compute_team_qb_features(sample_plays_with_qb)
        
        # EPA per dropback should be between -2.0 and 2.0 (typical range: -0.5 to 0.5)
        epa_values = qb_features_df["qb_epa_per_dropback"].dropna()
        assert all(epa_values >= -2.0)
        assert all(epa_values <= 2.0)
    
    def test_air_yards_if_available(self):
        """Test that air yards are computed if available."""
        plays_with_air_yards = pd.DataFrame({
            "game_id": ["nfl_2023_01_TEAM_OPP"] * 3,
            "play_id": [1, 2, 3],
            "posteam": ["TEAM"] * 3,
            "defteam": ["OPP"] * 3,
            "is_pass": [1, 1, 1],
            "passer_id": ["QB1", "QB1", "QB1"],
            "epa": [0.5, 0.3, 0.4],
            "success": [1, 1, 1],
            "air_yards": [10, 8, 12],  # Total: 30, per attempt: 10
        })
        
        metrics = compute_qb_metrics(plays_with_air_yards, "nfl_2023_01_TEAM_OPP", "TEAM", "QB1")
        
        assert abs(metrics["qb_air_yards_per_attempt"] - 10.0) < 0.01



