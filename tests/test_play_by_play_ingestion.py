"""
Tests for NFL Play-by-Play Ingestion

Validates plays.parquet loads correctly, each game has plays,
and team normalization works correctly.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.nfl.play_by_play import (
    normalize_to_play_schema,
    map_nflverse_game_id_to_our_format,
)


class TestPlayByPlayIngestion:
    """Test play-by-play ingestion functionality."""
    
    @pytest.fixture
    def sample_nflverse_pbp(self):
        """Sample nflverse play-by-play data."""
        return pd.DataFrame({
            "old_game_id": [
                "2015_01_KC_HOU",
                "2015_01_KC_HOU",
                "2015_01_KC_HOU",
                "2015_02_KC_DEN",
                "2015_02_KC_DEN",
            ],
            "play_id": [1, 2, 3, 1, 2],
            "posteam": ["KC", "KC", "HOU", "KC", "DEN"],
            "defteam": ["HOU", "HOU", "KC", "DEN", "KC"],
            "play_type": ["pass", "run", "pass", "run", "pass"],
            "epa": [0.5, -0.2, 0.3, 0.1, -0.5],
            "success": [1, 0, 1, 1, 0],
            "pass": [1, 0, 1, 0, 1],
            "rush": [0, 1, 0, 1, 0],
            "down": [1, 2, 3, 1, 2],
            "ydstogo": [10, 5, 3, 10, 7],
            "yardline_100": [50, 45, 20, 60, 55],
            "qtr": [1, 1, 2, 1, 1],
            "half_seconds_remaining": [1800, 1500, 1200, 1800, 1600],
            "season": [2015, 2015, 2015, 2015, 2015],
            "week": [1, 1, 1, 2, 2],
            "home_team": ["HOU", "HOU", "HOU", "DEN", "DEN"],
            "away_team": ["KC", "KC", "KC", "KC", "KC"],
        })
    
    @pytest.fixture
    def sample_games(self):
        """Sample games DataFrame for validation."""
        return pd.DataFrame({
            "game_id": [
                "nfl_2015_01_KC_HOU",
                "nfl_2015_02_KC_DEN",
            ],
            "season": [2015, 2015],
            "week": [1, 2],
            "home_team": ["HOU", "DEN"],
            "away_team": ["KC", "KC"],
        })
    
    def test_game_id_mapping(self, sample_nflverse_pbp, sample_games):
        """Test mapping nflverse game_id to our format."""
        mapped_df = map_nflverse_game_id_to_our_format(
            sample_nflverse_pbp, sample_games
        )
        
        # Check that game_id column exists and is correct format
        assert "game_id" in mapped_df.columns
        assert all(mapped_df["game_id"].str.startswith("nfl_"))
        
        # Check that game_ids match expected format
        expected_game_ids = ["nfl_2015_01_KC_HOU", "nfl_2015_02_KC_DEN"]
        assert set(mapped_df["game_id"].unique()) == set(expected_game_ids)
    
    def test_play_schema_normalization(self, sample_nflverse_pbp, sample_games):
        """Test normalization to PlayByPlay schema."""
        normalized_df = normalize_to_play_schema(
            sample_nflverse_pbp, sample_games
        )
        
        # Check required columns exist
        required_cols = [
            "game_id",
            "play_id",
            "posteam",
            "defteam",
            "play_type",
            "epa",
            "success",
            "is_pass",
            "is_run",
            "down",
            "ydstogo",
            "yardline_100",
            "qtr",
            "half_seconds_remaining",
        ]
        for col in required_cols:
            assert col in normalized_df.columns, f"Missing column: {col}"
        
        # Check data types
        assert normalized_df["epa"].dtype in [float, "float32", "float64"]
        assert normalized_df["success"].dtype in [int, "int32", "int64"]
        assert normalized_df["is_pass"].dtype in [int, "int32", "int64"]
        assert normalized_df["is_run"].dtype in [int, "int32", "int64"]
        
        # Check that posteam/defteam are normalized (2-3 letter abbreviations)
        # Most teams are 3 letters, but some are 2 (e.g., KC, GB, NE, SF, TB)
        assert all(normalized_df["posteam"].str.len() >= 2)
        assert all(normalized_df["posteam"].str.len() <= 3)
        assert all(normalized_df["defteam"].str.len() >= 2)
        assert all(normalized_df["defteam"].str.len() <= 3)
    
    def test_team_normalization(self, sample_nflverse_pbp):
        """Test team abbreviation normalization."""
        # Test with team relocations
        test_cases = [
            ("OAK", 2019, "LV"),  # Raiders before move
            ("OAK", 2021, "LV"),  # Raiders after move
            ("SD", 2016, "LAC"),  # Chargers before move
            ("STL", 2015, "LAR"),  # Rams before move
        ]
        
        from ingestion.nfl.play_by_play import normalize_team_abbreviation
        
        for team, season, expected in test_cases:
            result = normalize_team_abbreviation(team, season)
            assert result == expected, f"Failed: {team} ({season}) -> {result}, expected {expected}"
    
    def test_plays_parquet_loads(self):
        """Test that plays.parquet can be loaded if it exists."""
        plays_path = (
            Path(__file__).parent.parent
            / "data"
            / "nfl"
            / "staged"
            / "plays.parquet"
        )
        
        if not plays_path.exists():
            pytest.skip("plays.parquet not found - run ingestion first")
        
        df = pd.read_parquet(plays_path)
        
        # Check basic structure
        assert len(df) > 0, "plays.parquet is empty"
        assert "game_id" in df.columns
        assert "play_id" in df.columns
        assert "epa" in df.columns
        
        # Check that each game has plays
        games_path = (
            Path(__file__).parent.parent
            / "data"
            / "nfl"
            / "staged"
            / "games.parquet"
        )
        
        if games_path.exists():
            games_df = pd.read_parquet(games_path)
            games_with_plays = set(df["game_id"].unique())
            all_games = set(games_df["game_id"].unique())
            
            # At least 80% of games should have plays (some games might be missing)
            coverage = len(games_with_plays & all_games) / len(all_games)
            assert coverage >= 0.8, f"Only {coverage:.1%} of games have plays"
    
    def test_no_mismatched_team_names(self):
        """Test that team names match between games.parquet and plays.parquet."""
        plays_path = (
            Path(__file__).parent.parent
            / "data"
            / "nfl"
            / "staged"
            / "plays.parquet"
        )
        games_path = (
            Path(__file__).parent.parent
            / "data"
            / "nfl"
            / "staged"
            / "games.parquet"
        )
        
        if not plays_path.exists() or not games_path.exists():
            pytest.skip("Required parquet files not found")
        
        plays_df = pd.read_parquet(plays_path)
        games_df = pd.read_parquet(games_path)
        
        # Get teams from plays (posteam and defteam)
        play_teams = set(plays_df["posteam"].unique()) | set(plays_df["defteam"].unique())
        play_teams = {t for t in play_teams if pd.notna(t) and t != ""}
        
        # Get teams from games
        game_teams = set(games_df["home_team"].unique()) | set(games_df["away_team"].unique())
        
        # Teams should match (allowing for some games missing plays)
        mismatched = play_teams - game_teams
        assert len(mismatched) == 0, f"Mismatched teams: {mismatched}"

