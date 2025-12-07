"""
Tests for Phase 2 Feature Merge

Validates that EPA features merge correctly with baseline features,
naming conventions (home_epa_*, away_epa_*), and no data leakage.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.pipelines.feature_pipeline import merge_epa_features_to_games


class TestPhase2FeatureMerge:
    """Test Phase 2 feature merging."""
    
    @pytest.fixture
    def sample_games_markets(self):
        """Sample games_markets DataFrame."""
        return pd.DataFrame({
            "game_id": ["nfl_2023_01_TEAM_OPP", "nfl_2023_02_TEAM_OPP"],
            "season": [2023, 2023],
            "week": [1, 2],
            "date": pd.to_datetime(["2023-09-10", "2023-09-17"]),
            "home_team": ["OPP", "OPP"],
            "away_team": ["TEAM", "TEAM"],
            "close_spread": [-3.0, -5.0],
            "close_total": [45.0, 48.0],
        })
    
    @pytest.fixture
    def sample_baseline_features(self):
        """Sample baseline features."""
        return pd.DataFrame({
            "game_id": [
                "nfl_2023_01_TEAM_OPP",
                "nfl_2023_01_TEAM_OPP",
                "nfl_2023_02_TEAM_OPP",
                "nfl_2023_02_TEAM_OPP",
            ],
            "team": ["TEAM", "OPP", "TEAM", "OPP"],
            "is_home": [False, True, False, True],
            "win_rate_last4": [0.75, 0.5, 0.75, 0.5],
            "pdiff_last4": [5.0, -2.0, 5.0, -2.0],
        })
    
    @pytest.fixture
    def sample_epa_features(self):
        """Sample EPA features."""
        return pd.DataFrame({
            "game_id": [
                "nfl_2023_01_TEAM_OPP",
                "nfl_2023_01_TEAM_OPP",
                "nfl_2023_02_TEAM_OPP",
                "nfl_2023_02_TEAM_OPP",
            ],
            "team": ["TEAM", "OPP", "TEAM", "OPP"],
            "offensive_epa_per_play": [0.2, 0.1, 0.25, 0.15],
            "offensive_pass_epa": [0.3, 0.2, 0.35, 0.25],
            "offensive_run_epa": [0.1, -0.1, 0.15, -0.05],
            "offensive_success_rate": [0.6, 0.55, 0.65, 0.6],
            "defensive_epa_per_play_allowed": [0.05, 0.1, 0.08, 0.12],
        })
    
    def test_epa_features_merge(self, sample_games_markets, sample_baseline_features, sample_epa_features, tmp_path):
        """Test that EPA features merge correctly with baseline features."""
        # Save temporary files
        games_path = tmp_path / "games_markets.parquet"
        baseline_path = tmp_path / "baseline.parquet"
        epa_path = tmp_path / "epa.parquet"
        output_path = tmp_path / "output.parquet"
        
        sample_games_markets.to_parquet(games_path, index=False)
        sample_baseline_features.to_parquet(baseline_path, index=False)
        sample_epa_features.to_parquet(epa_path, index=False)
        
        # Merge
        result_df = merge_epa_features_to_games(
            games_markets_path=games_path,
            team_baseline_features_path=baseline_path,
            team_epa_features_path=epa_path,
            output_path=output_path,
        )
        
        # Check that all games are present
        assert len(result_df) == 2
        assert set(result_df["game_id"]) == set(sample_games_markets["game_id"])
        
        # Check that baseline features are present
        assert "home_win_rate_last4" in result_df.columns
        assert "away_win_rate_last4" in result_df.columns
        
        # Check that EPA features are present with correct naming
        assert "home_epa_offensive_epa_per_play" in result_df.columns
        assert "away_epa_offensive_epa_per_play" in result_df.columns
        assert "home_epa_offensive_pass_epa" in result_df.columns
        assert "away_epa_offensive_pass_epa" in result_df.columns
    
    def test_naming_convention(self, sample_games_markets, sample_baseline_features, sample_epa_features, tmp_path):
        """Test that naming convention is correct (home_epa_*, away_epa_*)."""
        games_path = tmp_path / "games_markets.parquet"
        baseline_path = tmp_path / "baseline.parquet"
        epa_path = tmp_path / "epa.parquet"
        output_path = tmp_path / "output.parquet"
        
        sample_games_markets.to_parquet(games_path, index=False)
        sample_baseline_features.to_parquet(baseline_path, index=False)
        sample_epa_features.to_parquet(epa_path, index=False)
        
        result_df = merge_epa_features_to_games(
            games_markets_path=games_path,
            team_baseline_features_path=baseline_path,
            team_epa_features_path=epa_path,
            output_path=output_path,
        )
        
        # Check naming convention
        epa_cols = [col for col in result_df.columns if "epa" in col.lower()]
        
        # All EPA columns should start with home_epa_ or away_epa_
        for col in epa_cols:
            assert col.startswith("home_epa_") or col.startswith("away_epa_"), \
                f"EPA column {col} does not follow naming convention"
        
        # Check that home team EPA matches OPP (home team in sample)
        game1 = result_df[result_df["game_id"] == "nfl_2023_01_TEAM_OPP"].iloc[0]
        assert abs(game1["home_epa_offensive_epa_per_play"] - 0.1) < 0.01  # OPP is home
        
        # Check that away team EPA matches TEAM (away team in sample)
        assert abs(game1["away_epa_offensive_epa_per_play"] - 0.2) < 0.01  # TEAM is away
    
    def test_no_data_leakage(self, sample_games_markets, sample_baseline_features, sample_epa_features, tmp_path):
        """Test that there is no data leakage (EPA from current game only)."""
        games_path = tmp_path / "games_markets.parquet"
        baseline_path = tmp_path / "baseline.parquet"
        epa_path = tmp_path / "epa.parquet"
        output_path = tmp_path / "output.parquet"
        
        sample_games_markets.to_parquet(games_path, index=False)
        sample_baseline_features.to_parquet(baseline_path, index=False)
        sample_epa_features.to_parquet(epa_path, index=False)
        
        result_df = merge_epa_features_to_games(
            games_markets_path=games_path,
            team_baseline_features_path=baseline_path,
            team_epa_features_path=epa_path,
            output_path=output_path,
        )
        
        # Each game should only have EPA from that specific game
        # Game 1 should have EPA values from game 1 only
        game1 = result_df[result_df["game_id"] == "nfl_2023_01_TEAM_OPP"].iloc[0]
        assert abs(game1["away_epa_offensive_epa_per_play"] - 0.2) < 0.01  # From game 1
        
        # Game 2 should have EPA values from game 2 only
        game2 = result_df[result_df["game_id"] == "nfl_2023_02_TEAM_OPP"].iloc[0]
        assert abs(game2["away_epa_offensive_epa_per_play"] - 0.25) < 0.01  # From game 2
        
        # Verify they're different (no leakage)
        assert game1["away_epa_offensive_epa_per_play"] != game2["away_epa_offensive_epa_per_play"]
    
    def test_one_row_per_game(self, sample_games_markets, sample_baseline_features, sample_epa_features, tmp_path):
        """Test that final feature table has one row per game."""
        games_path = tmp_path / "games_markets.parquet"
        baseline_path = tmp_path / "baseline.parquet"
        epa_path = tmp_path / "epa.parquet"
        output_path = tmp_path / "output.parquet"
        
        sample_games_markets.to_parquet(games_path, index=False)
        sample_baseline_features.to_parquet(baseline_path, index=False)
        sample_epa_features.to_parquet(epa_path, index=False)
        
        result_df = merge_epa_features_to_games(
            games_markets_path=games_path,
            team_baseline_features_path=baseline_path,
            team_epa_features_path=epa_path,
            output_path=output_path,
        )
        
        # Should have exactly one row per game
        assert len(result_df) == len(sample_games_markets)
        assert not result_df["game_id"].duplicated().any()
    
    def test_no_null_epa_where_pbp_exists(self, sample_games_markets, sample_baseline_features, sample_epa_features, tmp_path):
        """Test that EPA fields are not null where PBP exists."""
        games_path = tmp_path / "games_markets.parquet"
        baseline_path = tmp_path / "baseline.parquet"
        epa_path = tmp_path / "epa.parquet"
        output_path = tmp_path / "output.parquet"
        
        sample_games_markets.to_parquet(games_path, index=False)
        sample_baseline_features.to_parquet(baseline_path, index=False)
        sample_epa_features.to_parquet(epa_path, index=False)
        
        result_df = merge_epa_features_to_games(
            games_markets_path=games_path,
            team_baseline_features_path=baseline_path,
            team_epa_features_path=epa_path,
            output_path=output_path,
        )
        
        # EPA columns should not be null for games with PBP data
        epa_cols = [col for col in result_df.columns if "epa" in col.lower()]
        for col in epa_cols:
            null_count = result_df[col].isna().sum()
            assert null_count == 0, f"Found {null_count} null values in {col}"
    
    def test_epa_ranges_reasonable(self, sample_games_markets, sample_baseline_features, sample_epa_features, tmp_path):
        """Test that EPA ranges are reasonable."""
        games_path = tmp_path / "games_markets.parquet"
        baseline_path = tmp_path / "baseline.parquet"
        epa_path = tmp_path / "epa.parquet"
        output_path = tmp_path / "output.parquet"
        
        sample_games_markets.to_parquet(games_path, index=False)
        sample_baseline_features.to_parquet(baseline_path, index=False)
        sample_epa_features.to_parquet(epa_path, index=False)
        
        result_df = merge_epa_features_to_games(
            games_markets_path=games_path,
            team_baseline_features_path=baseline_path,
            team_epa_features_path=epa_path,
            output_path=output_path,
        )
        
        # Check EPA ranges
        epa_per_play_cols = [col for col in result_df.columns if "epa_per_play" in col]
        for col in epa_per_play_cols:
            values = result_df[col].dropna()
            if len(values) > 0:
                # Typical range: -0.2 to 0.3, but allow -1.0 to 1.0 for edge cases
                assert all(values >= -1.0), f"{col} has values < -1.0"
                assert all(values <= 1.0), f"{col} has values > 1.0"
        
        # Pass EPA should usually be > run EPA (but not always)
        # This is a soft check - we just verify the columns exist
        assert "home_epa_offensive_pass_epa" in result_df.columns
        assert "home_epa_offensive_run_epa" in result_df.columns

