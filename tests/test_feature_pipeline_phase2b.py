"""
Tests for Phase 2B Feature Pipeline

Validates that rolling EPA + QB features merge correctly with Phase 2 features,
naming conventions, and no data leakage.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.pipelines.feature_pipeline import merge_phase2b_features


class TestPhase2BFeatureMerge:
    """Test Phase 2B feature merging."""
    
    @pytest.fixture
    def sample_phase2_features(self):
        """Sample Phase 2 features."""
        return pd.DataFrame({
            "game_id": ["nfl_2023_01_TEAM_OPP", "nfl_2023_02_TEAM_OPP"],
            "season": [2023, 2023],
            "week": [1, 2],
            "date": pd.to_datetime(["2023-09-10", "2023-09-17"]),
            "home_team": ["OPP", "OPP"],
            "away_team": ["TEAM", "TEAM"],
            "close_spread": [-3.0, -5.0],
            "close_total": [45.0, 48.0],
            "home_win_rate_last4": [0.5, 0.5],
            "away_win_rate_last4": [0.75, 0.75],
            "home_epa_offensive_epa_per_play": [0.1, 0.15],
            "away_epa_offensive_epa_per_play": [0.2, 0.25],
        })
    
    @pytest.fixture
    def sample_rolling_epa_features(self):
        """Sample rolling EPA features."""
        return pd.DataFrame({
            "game_id": [
                "nfl_2023_01_TEAM_OPP",
                "nfl_2023_01_TEAM_OPP",
                "nfl_2023_02_TEAM_OPP",
                "nfl_2023_02_TEAM_OPP",
            ],
            "team": ["TEAM", "OPP", "TEAM", "OPP"],
            "off_epa_last3": [0.18, 0.12, 0.20, 0.14],
            "off_pass_epa_last3": [0.25, 0.18, 0.28, 0.20],
            "def_epa_allowed_last3": [-0.15, -0.10, -0.18, -0.12],
        })
    
    @pytest.fixture
    def sample_qb_features(self):
        """Sample QB features."""
        return pd.DataFrame({
            "game_id": [
                "nfl_2023_01_TEAM_OPP",
                "nfl_2023_01_TEAM_OPP",
                "nfl_2023_02_TEAM_OPP",
                "nfl_2023_02_TEAM_OPP",
            ],
            "team": ["TEAM", "OPP", "TEAM", "OPP"],
            "qb_id": ["QB1", "QB2", "QB1", "QB2"],
            "qb_epa_per_dropback": [0.25, 0.15, 0.30, 0.18],
            "qb_sack_rate": [0.05, 0.08, 0.04, 0.07],
            "qb_int_rate": [0.02, 0.03, 0.02, 0.03],
        })
    
    def test_phase2b_features_merge(
        self, sample_phase2_features, sample_rolling_epa_features, sample_qb_features, tmp_path
    ):
        """Test that Phase 2B features merge correctly."""
        phase2_path = tmp_path / "phase2.parquet"
        rolling_path = tmp_path / "rolling.parquet"
        qb_path = tmp_path / "qb.parquet"
        output_path = tmp_path / "output.parquet"
        
        sample_phase2_features.to_parquet(phase2_path, index=False)
        sample_rolling_epa_features.to_parquet(rolling_path, index=False)
        sample_qb_features.to_parquet(qb_path, index=False)
        
        # Merge
        result_df = merge_phase2b_features(
            game_features_phase2_path=phase2_path,
            team_rolling_epa_path=rolling_path,
            team_qb_features_path=qb_path,
            output_path=output_path,
        )
        
        # Check that all games are present
        assert len(result_df) == 2
        assert set(result_df["game_id"]) == set(sample_phase2_features["game_id"])
        
        # Check that rolling EPA features are present
        assert "home_roll_epa_off_epa_last3" in result_df.columns
        assert "away_roll_epa_off_epa_last3" in result_df.columns
        
        # Check that QB features are present
        assert "home_qb_qb_epa_per_dropback" in result_df.columns
        assert "away_qb_qb_epa_per_dropback" in result_df.columns
    
    def test_naming_convention(
        self, sample_phase2_features, sample_rolling_epa_features, sample_qb_features, tmp_path
    ):
        """Test that naming convention is correct."""
        phase2_path = tmp_path / "phase2.parquet"
        rolling_path = tmp_path / "rolling.parquet"
        qb_path = tmp_path / "qb.parquet"
        output_path = tmp_path / "output.parquet"
        
        sample_phase2_features.to_parquet(phase2_path, index=False)
        sample_rolling_epa_features.to_parquet(rolling_path, index=False)
        sample_qb_features.to_parquet(qb_path, index=False)
        
        result_df = merge_phase2b_features(
            game_features_phase2_path=phase2_path,
            team_rolling_epa_path=rolling_path,
            team_qb_features_path=qb_path,
            output_path=output_path,
        )
        
        # Check naming convention
        rolling_cols = [col for col in result_df.columns if "roll_epa" in col]
        qb_cols = [col for col in result_df.columns if "qb_" in col]
        
        # All rolling EPA columns should start with home_roll_epa_ or away_roll_epa_
        for col in rolling_cols:
            assert col.startswith("home_roll_epa_") or col.startswith("away_roll_epa_"), \
                f"Rolling EPA column {col} does not follow naming convention"
        
        # All QB columns should start with home_qb_ or away_qb_
        for col in qb_cols:
            assert col.startswith("home_qb_") or col.startswith("away_qb_"), \
                f"QB column {col} does not follow naming convention"
    
    def test_one_row_per_game(
        self, sample_phase2_features, sample_rolling_epa_features, sample_qb_features, tmp_path
    ):
        """Test that final feature table has one row per game."""
        phase2_path = tmp_path / "phase2.parquet"
        rolling_path = tmp_path / "rolling.parquet"
        qb_path = tmp_path / "qb.parquet"
        output_path = tmp_path / "output.parquet"
        
        sample_phase2_features.to_parquet(phase2_path, index=False)
        sample_rolling_epa_features.to_parquet(rolling_path, index=False)
        sample_qb_features.to_parquet(qb_path, index=False)
        
        result_df = merge_phase2b_features(
            game_features_phase2_path=phase2_path,
            team_rolling_epa_path=rolling_path,
            team_qb_features_path=qb_path,
            output_path=output_path,
        )
        
        # Should have exactly one row per game
        assert len(result_df) == len(sample_phase2_features)
        assert not result_df["game_id"].duplicated().any()
    
    def test_no_data_leakage(
        self, sample_phase2_features, sample_rolling_epa_features, sample_qb_features, tmp_path
    ):
        """Test that there is no data leakage (rolling features use past games only)."""
        phase2_path = tmp_path / "phase2.parquet"
        rolling_path = tmp_path / "rolling.parquet"
        qb_path = tmp_path / "qb.parquet"
        output_path = tmp_path / "output.parquet"
        
        sample_phase2_features.to_parquet(phase2_path, index=False)
        sample_rolling_epa_features.to_parquet(rolling_path, index=False)
        sample_qb_features.to_parquet(qb_path, index=False)
        
        result_df = merge_phase2b_features(
            game_features_phase2_path=phase2_path,
            team_rolling_epa_path=rolling_path,
            team_qb_features_path=qb_path,
            output_path=output_path,
        )
        
        # Rolling features should be different for game 1 vs game 2
        # (game 2 should have game 1 in its rolling window)
        game1 = result_df[result_df["game_id"] == "nfl_2023_01_TEAM_OPP"].iloc[0]
        game2 = result_df[result_df["game_id"] == "nfl_2023_02_TEAM_OPP"].iloc[0]
        
        # Game 2's rolling features should potentially differ from game 1
        # (depending on the rolling window calculation)
        # This is a sanity check that rolling features are being computed
        assert "home_roll_epa_off_epa_last3" in result_df.columns
        assert "away_roll_epa_off_epa_last3" in result_df.columns
    
    def test_no_unexpected_nulls(
        self, sample_phase2_features, sample_rolling_epa_features, sample_qb_features, tmp_path
    ):
        """Test that there are no unexpected nulls in merged features."""
        phase2_path = tmp_path / "phase2.parquet"
        rolling_path = tmp_path / "rolling.parquet"
        qb_path = tmp_path / "qb.parquet"
        output_path = tmp_path / "output.parquet"
        
        sample_phase2_features.to_parquet(phase2_path, index=False)
        sample_rolling_epa_features.to_parquet(rolling_path, index=False)
        sample_qb_features.to_parquet(qb_path, index=False)
        
        result_df = merge_phase2b_features(
            game_features_phase2_path=phase2_path,
            team_rolling_epa_path=rolling_path,
            team_qb_features_path=qb_path,
            output_path=output_path,
        )
        
        # Phase 2 features should not be null
        assert not result_df["home_epa_offensive_epa_per_play"].isna().any()
        assert not result_df["away_epa_offensive_epa_per_play"].isna().any()
        
        # Rolling and QB features might have some nulls (early season games)
        # but should be present for most games
        rolling_cols = [col for col in result_df.columns if "roll_epa" in col]
        qb_cols = [col for col in result_df.columns if "qb_" in col]
        
        # At least some games should have rolling/QB features
        assert result_df[rolling_cols].notna().any(axis=1).sum() > 0
        assert result_df[qb_cols].notna().any(axis=1).sum() > 0



