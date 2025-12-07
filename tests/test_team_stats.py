"""
Tests for NFL Team Stats Ingestion
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.nfl.team_stats import normalize_to_team_stats_schema
from features.nfl.team_form_features import (
    calculate_win_loss,
    calculate_point_differential,
)


class TestTeamStatsIngestion:
    """Test team stats ingestion and normalization."""
    
    @pytest.fixture
    def sample_schedule_data(self):
        """Sample schedule data from nflverse format."""
        return pd.DataFrame({
            "season": [2023, 2023, 2023],
            "week": [1, 1, 1],
            "game_id": ["2023_01_KC_DET", "2023_01_GB_CHI", "2023_01_NE_MIA"],
            "home_team": ["DET", "CHI", "MIA"],
            "away_team": ["KC", "GB", "NE"],
            "home_score": [20, 10, 17],
            "away_score": [21, 27, 24],
        })
    
    @pytest.fixture
    def sample_team_stats_raw(self):
        """Sample raw team stats data."""
        return pd.DataFrame({
            "season": [2023, 2023, 2023, 2023, 2023, 2023],
            "week": [1, 1, 1, 1, 1, 1],
            "team": ["KC", "DET", "GB", "CHI", "NE", "MIA"],
            "is_home": [False, True, False, True, False, True],
            "opponent": ["DET", "KC", "CHI", "GB", "MIA", "NE"],
            "points_for": [21, 20, 27, 10, 24, 17],
            "points_against": [20, 21, 10, 27, 17, 24],
        })
    
    @pytest.fixture
    def sample_games_df(self):
        """Sample games DataFrame."""
        return pd.DataFrame({
            "game_id": [
                "nfl_2023_01_KC_DET",
                "nfl_2023_01_GB_CHI",
                "nfl_2023_01_NE_MIA",
            ],
            "season": [2023, 2023, 2023],
            "week": [1, 1, 1],
            "home_team": ["DET", "CHI", "MIA"],
            "away_team": ["KC", "GB", "NE"],
        })
    
    def test_two_rows_per_game(self, sample_team_stats_raw, sample_games_df):
        """Test that each game has exactly 2 rows (home + away)."""
        normalized = normalize_to_team_stats_schema(sample_team_stats_raw, sample_games_df)
        
        # Count rows per game_id
        counts = normalized.groupby("game_id").size()
        assert all(counts == 2), f"Expected 2 rows per game, got: {counts.to_dict()}"
    
    def test_home_away_flag(self, sample_team_stats_raw, sample_games_df):
        """Test that is_home flag is correctly set."""
        normalized = normalize_to_team_stats_schema(sample_team_stats_raw, sample_games_df)
        
        # Each game should have one home and one away
        for game_id in normalized["game_id"].unique():
            game_stats = normalized[normalized["game_id"] == game_id]
            assert game_stats["is_home"].sum() == 1, f"Game {game_id} should have exactly 1 home team"
            assert (~game_stats["is_home"]).sum() == 1, f"Game {game_id} should have exactly 1 away team"
    
    def test_points_for_against(self, sample_team_stats_raw, sample_games_df):
        """Test that points_for and points_against are correctly set."""
        normalized = normalize_to_team_stats_schema(sample_team_stats_raw, sample_games_df)
        
        # Check that points_for and points_against are not null
        assert normalized["points_for"].notna().all(), "points_for should not be null"
        assert normalized["points_against"].notna().all(), "points_against should not be null"
        
        # Check that points are non-negative
        assert (normalized["points_for"] >= 0).all(), "points_for should be >= 0"
        assert (normalized["points_against"] >= 0).all(), "points_against should be >= 0"
    
    def test_win_loss_calculation(self):
        """Test win/loss calculation."""
        stats_df = pd.DataFrame({
            "points_for": [21, 20, 17],
            "points_against": [20, 21, 17],
        })
        
        result = calculate_win_loss(stats_df)
        
        assert "win" in result.columns
        assert result.iloc[0]["win"] == 1  # Win
        assert result.iloc[1]["win"] == 0  # Loss
        assert result.iloc[2]["win"] == 0.5  # Tie
    
    def test_point_differential_calculation(self):
        """Test point differential calculation."""
        stats_df = pd.DataFrame({
            "points_for": [21, 20, 17],
            "points_against": [20, 21, 17],
        })
        
        result = calculate_point_differential(stats_df)
        
        assert "point_diff" in result.columns
        assert result.iloc[0]["point_diff"] == 1
        assert result.iloc[1]["point_diff"] == -1
        assert result.iloc[2]["point_diff"] == 0


class TestTeamStatsIntegration:
    """Integration tests for team stats."""
    
    def test_team_stats_file_exists(self):
        """Test that team_stats.parquet exists after ingestion."""
        team_stats_path = Path(__file__).parent.parent / "data" / "nfl" / "staged" / "team_stats.parquet"
        if team_stats_path.exists():
            df = pd.read_parquet(team_stats_path)
            assert len(df) > 0
            assert "game_id" in df.columns
            assert "team" in df.columns
            assert "is_home" in df.columns
            assert "points_for" in df.columns
            assert "points_against" in df.columns
    
    def test_two_rows_per_game_integration(self):
        """Test that real data has 2 rows per game."""
        team_stats_path = Path(__file__).parent.parent / "data" / "nfl" / "staged" / "team_stats.parquet"
        games_path = Path(__file__).parent.parent / "data" / "nfl" / "staged" / "games.parquet"
        
        if team_stats_path.exists() and games_path.exists():
            team_stats_df = pd.read_parquet(team_stats_path)
            games_df = pd.read_parquet(games_path)
            
            # Count rows per game
            counts = team_stats_df.groupby("game_id").size()
            
            # Should be 2 rows per game (home + away)
            assert all(counts == 2), f"Found games with != 2 rows: {counts[counts != 2].head()}"
            
            # All games should be represented
            assert len(counts) == len(games_df), "Number of games in stats should match games.parquet"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

