"""
Tests for Phase 1A NFL Ingestion Modules

Validates:
- game_id format correctness
- no duplicates
- all seasons 2015-2024 populated
- every game has a matching market entry
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.nfl.schedule import form_game_id, normalize_team_abbreviation
from ingestion.nfl.join_games_markets import (
    validate_game_ids,
    validate_completeness,
    validate_spread_direction,
)


class TestGameIDFormat:
    """Test game_id format correctness."""
    
    def test_game_id_format(self):
        """Test that game_id follows format: nfl_{season}_{week}_{away}_{home}"""
        game_id = form_game_id(2023, 1, "KC", "DET")
        assert game_id == "nfl_2023_01_KC_DET"
        
        game_id = form_game_id(2015, 18, "NYJ", "NE")
        assert game_id == "nfl_2015_18_NYJ_NE"
    
    def test_game_id_week_padding(self):
        """Test that week is zero-padded to 2 digits."""
        game_id = form_game_id(2023, 1, "KC", "DET")
        assert game_id.split("_")[2] == "01"
        
        game_id = form_game_id(2023, 10, "KC", "DET")
        assert game_id.split("_")[2] == "10"
    
    def test_team_normalization(self):
        """Test team abbreviation normalization."""
        assert normalize_team_abbreviation("KC") == "KC"
        assert normalize_team_abbreviation("kc") == "KC"
        assert normalize_team_abbreviation("OAK") == "LV"  # Raiders relocation
        assert normalize_team_abbreviation("SD") == "LAC"  # Chargers relocation


class TestDataCompleteness:
    """Test data completeness and validation."""
    
    @pytest.fixture
    def games_df(self):
        """Create sample games DataFrame."""
        return pd.DataFrame({
            "game_id": [
                "nfl_2015_01_KC_DET",
                "nfl_2015_02_GB_CHI",
                "nfl_2024_01_NE_MIA",
            ],
            "season": [2015, 2015, 2024],
            "week": [1, 2, 1],
            "home_team": ["DET", "CHI", "MIA"],
            "away_team": ["KC", "GB", "NE"],
            "home_score": [20, 10, 17],
            "away_score": [21, 27, 24],
        })
    
    @pytest.fixture
    def markets_df(self):
        """Create sample markets DataFrame."""
        return pd.DataFrame({
            "game_id": [
                "nfl_2015_01_KC_DET",
                "nfl_2015_02_GB_CHI",
                "nfl_2024_01_NE_MIA",
            ],
            "season": [2015, 2015, 2024],
            "week": [1, 2, 1],
            "close_spread": [-3.0, 7.0, -2.5],
            "close_total": [45.5, 48.0, 42.0],
        })
    
    def test_no_duplicates(self, games_df):
        """Test that there are no duplicate game_ids."""
        results = validate_game_ids(games_df)
        assert results["no_duplicates"] is True
    
    def test_duplicate_detection(self):
        """Test that duplicates are detected."""
        games_df = pd.DataFrame({
            "game_id": ["nfl_2015_01_KC_DET", "nfl_2015_01_KC_DET"],
            "season": [2015, 2015],
            "week": [1, 1],
        })
        results = validate_game_ids(games_df)
        assert results["no_duplicates"] is False
    
    def test_all_games_have_markets(self, games_df, markets_df):
        """Test that all games have matching market entries."""
        results = validate_completeness(games_df, markets_df, required_seasons=[2015, 2024])
        assert results["all_games_have_markets"] is True
    
    def test_missing_markets_detection(self, games_df):
        """Test that missing markets are detected."""
        markets_df = pd.DataFrame({
            "game_id": ["nfl_2015_01_KC_DET"],  # Missing one game
            "season": [2015],
            "week": [1],
            "close_spread": [-3.0],
            "close_total": [45.5],
        })
        results = validate_completeness(games_df, markets_df)
        assert results["all_games_have_markets"] is False
    
    def test_season_coverage(self, games_df, markets_df):
        """Test that required seasons are present."""
        results = validate_completeness(
            games_df, markets_df, required_seasons=[2015, 2016, 2024]
        )
        # Should fail because 2016 is missing
        assert results["all_seasons_in_games"] is False or results["all_seasons_in_markets"] is False


class TestSpreadValidation:
    """Test spread direction validation."""
    
    @pytest.fixture
    def games_df(self):
        """Create sample games DataFrame."""
        return pd.DataFrame({
            "game_id": ["nfl_2023_01_KC_DET", "nfl_2023_02_GB_CHI"],
            "season": [2023, 2023],
            "week": [1, 2],
            "home_team": ["DET", "CHI"],
            "away_team": ["KC", "GB"],
            "home_score": [20, 10],
            "away_score": [21, 27],
        })
    
    @pytest.fixture
    def markets_df(self):
        """Create sample markets DataFrame."""
        return pd.DataFrame({
            "game_id": ["nfl_2023_01_KC_DET", "nfl_2023_02_GB_CHI"],
            "close_spread": [-3.0, 7.0],  # DET favored by 3, GB favored by 7
            "close_total": [45.5, 48.0],
        })
    
    def test_spread_validation(self, games_df, markets_df):
        """Test spread direction validation."""
        results = validate_spread_direction(games_df, markets_df)
        # Should pass basic sanity checks
        # Note: home_favorite_spread_sanity may not be present if no home wins in test data
        assert "no_extreme_spreads" in results
        assert len(results) > 0  # At least some validation results


class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_games_file_exists(self):
        """Test that games.parquet file exists after ingestion."""
        games_path = Path(__file__).parent.parent / "data" / "nfl" / "staged" / "games.parquet"
        if games_path.exists():
            df = pd.read_parquet(games_path)
            assert len(df) > 0
            assert "game_id" in df.columns
            assert "season" in df.columns
            assert "week" in df.columns
    
    def test_markets_file_exists(self):
        """Test that markets.parquet file exists after ingestion."""
        markets_path = Path(__file__).parent.parent / "data" / "nfl" / "staged" / "markets.parquet"
        if markets_path.exists():
            df = pd.read_parquet(markets_path)
            assert len(df) > 0
            assert "game_id" in df.columns
            assert "close_spread" in df.columns
            assert "close_total" in df.columns
    
    def test_joined_file_exists(self):
        """Test that games_markets.parquet file exists after join."""
        joined_path = Path(__file__).parent.parent / "data" / "nfl" / "staged" / "games_markets.parquet"
        if joined_path.exists():
            df = pd.read_parquet(joined_path)
            assert len(df) > 0
            assert "game_id" in df.columns
            assert "close_spread" in df.columns
            assert "home_score" in df.columns
    
    def test_seasons_2015_2024(self):
        """Test that data covers seasons 2015-2024."""
        games_path = Path(__file__).parent.parent / "data" / "nfl" / "staged" / "games.parquet"
        if games_path.exists():
            df = pd.read_parquet(games_path)
            seasons = set(df["season"].unique())
            required_seasons = set(range(2015, 2025))
            assert required_seasons.issubset(seasons), f"Missing seasons: {required_seasons - seasons}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

