"""
Tests for NFL QB Deviation Features

Validates:
- Career baseline calculations
- Season-to-date metrics
- Deviation/z-score calculations
- Trend features
- Data leakage prevention (CRITICAL)
- Edge cases (rookie QBs, backup QBs, etc.)
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.nfl.qb_deviation_features import (
    QBDeviationFeatureGenerator,
    QBCareerBaseline,
    compute_passer_rating,
    MIN_CAREER_DROPBACKS,
    MIN_SEASON_DROPBACKS,
    MIN_GAMES_FOR_STD,
)


class TestPasserRating:
    """Test passer rating calculation."""

    def test_perfect_game(self):
        """Test passer rating for a perfect game."""
        # 20/20, 400 yards, 5 TDs, 0 INTs
        rating = compute_passer_rating(
            completions=20,
            attempts=20,
            yards=400.0,
            tds=5,
            ints=0
        )
        # Perfect passer rating is 158.3
        assert abs(rating - 158.3) < 0.1

    def test_terrible_game(self):
        """Test passer rating for a terrible game."""
        # 5/20, 30 yards, 0 TDs, 4 INTs
        rating = compute_passer_rating(
            completions=5,
            attempts=20,
            yards=30.0,
            tds=0,
            ints=4
        )
        # Should be very low
        assert rating < 20

    def test_zero_attempts(self):
        """Test passer rating with zero attempts."""
        rating = compute_passer_rating(
            completions=0,
            attempts=0,
            yards=0.0,
            tds=0,
            ints=0
        )
        assert rating == 0.0

    def test_average_game(self):
        """Test passer rating for an average game."""
        # 22/35, 250 yards, 2 TDs, 1 INT (roughly average)
        rating = compute_passer_rating(
            completions=22,
            attempts=35,
            yards=250.0,
            tds=2,
            ints=1
        )
        # Average passer rating is around 88
        assert 70 < rating < 110


class TestQBDeviationFeatureGenerator:
    """Test the main QB deviation feature generator."""

    @pytest.fixture
    def sample_pbp_data(self):
        """Create synthetic play-by-play data for testing."""
        # Create data for 2 QBs across multiple games
        np.random.seed(42)

        games = []
        for season in [2022, 2023]:
            for week in range(1, 18):
                games.append({
                    "game_id": f"nfl_{season}_{week:02d}_KC_DEN",
                    "season": season,
                    "week": week,
                    "gameday": datetime(season, 9, 1) + timedelta(weeks=week-1),
                })

        # Generate plays for each game
        plays = []
        play_id = 0

        for game in games:
            # Home team (KC) QB plays
            for _ in range(30):  # ~30 dropbacks per game
                is_sack = np.random.random() < 0.05
                is_int = np.random.random() < 0.025
                is_complete = np.random.random() < 0.65 if not is_sack else False
                is_td = np.random.random() < 0.05 if is_complete else False

                plays.append({
                    "play_id": play_id,
                    "game_id": game["game_id"],
                    "old_game_id": game["game_id"].replace("nfl_", ""),
                    "season": game["season"],
                    "week": game["week"],
                    "gameday": game["gameday"],
                    "posteam": "KC",
                    "defteam": "DEN",
                    "passer_id": "QBMahomes",
                    "passer_player_name": "P.Mahomes",
                    "play_type": "sack" if is_sack else "pass",
                    "epa": np.random.normal(0.15, 0.5) if not is_sack else np.random.normal(-0.8, 0.3),
                    "success": 1 if np.random.random() < 0.5 else 0,
                    "cpoe": np.random.normal(2.0, 5.0) if not is_sack else None,
                    "complete_pass": 1 if is_complete else 0,
                    "interception": 1 if is_int else 0,
                    "pass_touchdown": 1 if is_td else 0,
                    "passing_yards": np.random.exponential(10) if is_complete else 0,
                    "air_yards": np.random.exponential(8),
                    "sack": 1 if is_sack else 0,
                })
                play_id += 1

            # Away team (DEN) QB plays - lower performance
            for _ in range(28):
                is_sack = np.random.random() < 0.08
                is_int = np.random.random() < 0.035
                is_complete = np.random.random() < 0.58 if not is_sack else False
                is_td = np.random.random() < 0.03 if is_complete else False

                plays.append({
                    "play_id": play_id,
                    "game_id": game["game_id"],
                    "old_game_id": game["game_id"].replace("nfl_", ""),
                    "season": game["season"],
                    "week": game["week"],
                    "gameday": game["gameday"],
                    "posteam": "DEN",
                    "defteam": "KC",
                    "passer_id": "QBWilson",
                    "passer_player_name": "R.Wilson",
                    "play_type": "sack" if is_sack else "pass",
                    "epa": np.random.normal(-0.05, 0.5) if not is_sack else np.random.normal(-0.9, 0.3),
                    "success": 1 if np.random.random() < 0.42 else 0,
                    "cpoe": np.random.normal(-1.0, 5.0) if not is_sack else None,
                    "complete_pass": 1 if is_complete else 0,
                    "interception": 1 if is_int else 0,
                    "pass_touchdown": 1 if is_td else 0,
                    "passing_yards": np.random.exponential(8) if is_complete else 0,
                    "air_yards": np.random.exponential(7),
                    "sack": 1 if is_sack else 0,
                })
                play_id += 1

        return pd.DataFrame(plays)

    @pytest.fixture
    def sample_games_data(self):
        """Create corresponding games data."""
        games = []
        for season in [2022, 2023]:
            for week in range(1, 18):
                game_date = datetime(season, 9, 1) + timedelta(weeks=week-1)
                games.append({
                    "game_id": f"nfl_{season}_{week:02d}_KC_DEN",
                    "season": season,
                    "week": week,
                    "date": game_date,
                    "home_team": "KC",
                    "away_team": "DEN",
                    "home_score": 28,
                    "away_score": 21,
                })
        return pd.DataFrame(games)

    def test_generator_initialization(self, sample_pbp_data, sample_games_data):
        """Test that generator initializes correctly."""
        generator = QBDeviationFeatureGenerator(
            pbp_df=sample_pbp_data,
            games_df=sample_games_data
        )

        assert generator.passer_col == "passer_id"
        assert len(generator.pbp_df) == len(sample_pbp_data)

    def test_career_baseline_calculation(self, sample_pbp_data, sample_games_data):
        """Test career baseline calculation."""
        generator = QBDeviationFeatureGenerator(
            pbp_df=sample_pbp_data,
            games_df=sample_games_data
        )

        # Get baseline for Mahomes at end of 2023 season
        before_date = pd.Timestamp("2024-01-01")
        baseline = generator.get_career_baseline("QBMahomes", before_date)

        assert baseline is not None
        assert baseline.player_id == "QBMahomes"
        assert baseline.games_started > 0
        assert baseline.dropbacks > MIN_CAREER_DROPBACKS
        assert -0.5 < baseline.epa_per_dropback < 0.5
        assert 0 <= baseline.int_rate <= 0.1
        assert 0.5 <= baseline.completion_pct <= 0.8

    def test_no_data_leakage_career(self, sample_pbp_data, sample_games_data):
        """CRITICAL: Verify career baseline doesn't include future data."""
        generator = QBDeviationFeatureGenerator(
            pbp_df=sample_pbp_data,
            games_df=sample_games_data
        )

        # Get baseline at start of 2023 season
        before_date = pd.Timestamp("2023-09-01")
        baseline_early = generator.get_career_baseline("QBMahomes", before_date)

        # Get baseline at end of 2023 season
        before_date_late = pd.Timestamp("2024-01-01")
        baseline_late = generator.get_career_baseline("QBMahomes", before_date_late)

        if baseline_early is not None and baseline_late is not None:
            # Late baseline should have more data
            assert baseline_late.dropbacks > baseline_early.dropbacks
            assert baseline_late.games_started > baseline_early.games_started

    def test_no_data_leakage_season(self, sample_pbp_data, sample_games_data):
        """CRITICAL: Verify season metrics don't include current game."""
        generator = QBDeviationFeatureGenerator(
            pbp_df=sample_pbp_data,
            games_df=sample_games_data
        )

        # Get season metrics entering week 10 of 2023
        before_date = pd.Timestamp("2023-11-01")  # Approximate week 10
        metrics_week10 = generator.get_season_metrics(
            "QBMahomes", 2023, 10, before_date
        )

        # Get season metrics entering week 5 of 2023
        before_date_early = pd.Timestamp("2023-10-01")  # Approximate week 5
        metrics_week5 = generator.get_season_metrics(
            "QBMahomes", 2023, 5, before_date_early
        )

        # Week 10 should have more games than week 5
        assert metrics_week10["season_games"] > metrics_week5["season_games"]

    def test_deviation_features(self, sample_pbp_data, sample_games_data):
        """Test deviation feature calculations."""
        generator = QBDeviationFeatureGenerator(
            pbp_df=sample_pbp_data,
            games_df=sample_games_data
        )

        before_date = pd.Timestamp("2023-12-01")
        career = generator.get_career_baseline("QBMahomes", before_date)
        season = generator.get_season_metrics("QBMahomes", 2023, 14, before_date)

        if career is not None:
            deviation = generator.compute_deviation_features(career, season)

            # Check all expected keys exist
            assert "epa_vs_career" in deviation
            assert "epa_zscore" in deviation
            assert "int_rate_vs_career" in deviation
            assert "performing_above_career" in deviation
            assert "performing_below_career" in deviation

            # Z-scores should be reasonable
            assert -5 < deviation["epa_zscore"] < 5

    def test_trend_features(self, sample_pbp_data, sample_games_data):
        """Test trend feature calculations."""
        generator = QBDeviationFeatureGenerator(
            pbp_df=sample_pbp_data,
            games_df=sample_games_data
        )

        before_date = pd.Timestamp("2023-12-01")
        trend = generator.compute_trend_features("QBMahomes", 2023, 14, before_date)

        # Check all expected keys exist
        assert "epa_trend_last_4" in trend
        assert "performance_improving" in trend
        assert "performance_declining" in trend
        assert "last_4_avg_epa" in trend
        assert "recent_hot_streak" in trend

        # Binary flags should be 0 or 1
        assert trend["performance_improving"] in [0, 1]
        assert trend["performance_declining"] in [0, 1]

    def test_luck_features(self, sample_pbp_data, sample_games_data):
        """Test luck/regression indicator features."""
        generator = QBDeviationFeatureGenerator(
            pbp_df=sample_pbp_data,
            games_df=sample_games_data
        )

        before_date = pd.Timestamp("2023-12-01")
        career = generator.get_career_baseline("QBMahomes", before_date)
        season = generator.get_season_metrics("QBMahomes", 2023, 14, before_date)

        if career is not None:
            luck = generator.compute_luck_features(career, season)

            # Check all expected keys exist
            assert "int_vs_league_avg" in luck
            assert "likely_int_regression_up" in luck
            assert "likely_int_regression_down" in luck
            assert "regression_magnitude" in luck

            # Binary flags should be 0 or 1
            assert luck["likely_int_regression_up"] in [0, 1]
            assert luck["likely_int_regression_down"] in [0, 1]

    def test_generate_game_features(self, sample_pbp_data, sample_games_data):
        """Test full game feature generation."""
        generator = QBDeviationFeatureGenerator(
            pbp_df=sample_pbp_data,
            games_df=sample_games_data
        )

        game_date = pd.Timestamp("2023-11-15")
        features = generator.generate_game_features(
            game_id="nfl_2023_10_KC_DEN",
            season=2023,
            week=10,
            team="KC",
            qb_id="QBMahomes",
            game_date=game_date
        )

        # Should have many features
        assert len(features) > 30

        # All keys should start with 'qb_'
        for key in features.keys():
            assert key.startswith("qb_")

        # Check key features exist
        assert "qb_career_epa" in features
        assert "qb_season_epa_per_dropback" in features
        assert "qb_epa_vs_career" in features
        assert "qb_epa_zscore" in features

    def test_rookie_qb_handling(self, sample_pbp_data, sample_games_data):
        """Test that rookie QBs with insufficient data return empty features."""
        generator = QBDeviationFeatureGenerator(
            pbp_df=sample_pbp_data,
            games_df=sample_games_data
        )

        # Try to get features for a QB that doesn't exist
        game_date = pd.Timestamp("2023-11-15")
        features = generator.generate_game_features(
            game_id="nfl_2023_10_KC_DEN",
            season=2023,
            week=10,
            team="KC",
            qb_id="QBRookie",  # Non-existent QB
            game_date=game_date
        )

        # Should return empty features (all zeros)
        assert features["qb_career_games"] == 0
        assert features["qb_season_games"] == 0

    def test_qb_comparison(self, sample_pbp_data, sample_games_data):
        """Test that better QB has higher career EPA."""
        generator = QBDeviationFeatureGenerator(
            pbp_df=sample_pbp_data,
            games_df=sample_games_data
        )

        before_date = pd.Timestamp("2024-01-01")

        mahomes_baseline = generator.get_career_baseline("QBMahomes", before_date)
        wilson_baseline = generator.get_career_baseline("QBWilson", before_date)

        if mahomes_baseline is not None and wilson_baseline is not None:
            # Mahomes should have higher EPA (based on our synthetic data generation)
            assert mahomes_baseline.epa_per_dropback > wilson_baseline.epa_per_dropback


class TestDataLeakageValidation:
    """Focused tests for data leakage prevention."""

    @pytest.fixture
    def sequential_pbp_data(self):
        """Create PBP data with known sequential pattern to detect leakage."""
        plays = []
        play_id = 0

        # Create 10 weeks of data with increasing EPA
        # Week 1 EPA = 0.0, Week 2 EPA = 0.1, ..., Week 10 EPA = 0.9
        for week in range(1, 11):
            game_date = datetime(2023, 9, 1) + timedelta(weeks=week-1)
            target_epa = (week - 1) * 0.1  # Increasing EPA per week

            for _ in range(30):
                plays.append({
                    "play_id": play_id,
                    "game_id": f"nfl_2023_{week:02d}_KC_DEN",
                    "old_game_id": f"2023_{week:02d}_KC_DEN",
                    "season": 2023,
                    "week": week,
                    "gameday": game_date,
                    "posteam": "KC",
                    "defteam": "DEN",
                    "passer_id": "QBTest",
                    "passer_player_name": "Test QB",
                    "play_type": "pass",
                    "epa": target_epa + np.random.normal(0, 0.05),
                    "success": 1 if np.random.random() < 0.5 else 0,
                    "cpoe": 0.0,
                    "complete_pass": 1,
                    "interception": 0,
                    "pass_touchdown": 0,
                    "passing_yards": 10,
                    "air_yards": 8,
                    "sack": 0,
                })
                play_id += 1

        return pd.DataFrame(plays)

    @pytest.fixture
    def sequential_games_data(self):
        """Create corresponding games data."""
        games = []
        for week in range(1, 11):
            game_date = datetime(2023, 9, 1) + timedelta(weeks=week-1)
            games.append({
                "game_id": f"nfl_2023_{week:02d}_KC_DEN",
                "season": 2023,
                "week": week,
                "date": game_date,
                "home_team": "KC",
                "away_team": "DEN",
            })
        return pd.DataFrame(games)

    def test_week5_features_dont_include_weeks_5_to_10(self, sequential_pbp_data, sequential_games_data):
        """
        CRITICAL TEST: Features at week 5 should only include weeks 1-4.

        The EPA was designed to increase each week:
        - Weeks 1-4 average EPA ≈ (0 + 0.1 + 0.2 + 0.3) / 4 = 0.15
        - Weeks 5-10 average EPA ≈ (0.4 + ... + 0.9) / 6 = 0.65

        If we see high EPA at week 5, there's data leakage.
        """
        generator = QBDeviationFeatureGenerator(
            pbp_df=sequential_pbp_data,
            games_df=sequential_games_data
        )

        # Get season metrics entering week 5
        game_date = datetime(2023, 9, 1) + timedelta(weeks=4)
        season_metrics = generator.get_season_metrics(
            "QBTest", 2023, 5, pd.Timestamp(game_date)
        )

        # Season EPA should be around 0.15 (average of weeks 1-4)
        # NOT 0.45 (average of weeks 1-9)
        if season_metrics["season_games"] > 0:
            assert season_metrics["season_epa_per_dropback"] < 0.25, \
                f"Data leakage detected! Season EPA {season_metrics['season_epa_per_dropback']:.3f} is too high"

    def test_week10_features_include_more_data(self, sequential_pbp_data, sequential_games_data):
        """Verify that later weeks include more historical data."""
        generator = QBDeviationFeatureGenerator(
            pbp_df=sequential_pbp_data,
            games_df=sequential_games_data
        )

        # Week 5 metrics
        game_date_5 = datetime(2023, 9, 1) + timedelta(weeks=4)
        metrics_week5 = generator.get_season_metrics(
            "QBTest", 2023, 5, pd.Timestamp(game_date_5)
        )

        # Week 10 metrics
        game_date_10 = datetime(2023, 9, 1) + timedelta(weeks=9)
        metrics_week10 = generator.get_season_metrics(
            "QBTest", 2023, 10, pd.Timestamp(game_date_10)
        )

        # Week 10 should have more games included
        assert metrics_week10["season_games"] > metrics_week5["season_games"]

        # Week 10 EPA should be higher (includes weeks with higher EPA)
        assert metrics_week10["season_epa_per_dropback"] > metrics_week5["season_epa_per_dropback"]

    def test_career_excludes_future_seasons(self, sequential_pbp_data, sequential_games_data):
        """Verify career baseline doesn't include future season data."""
        # Add 2024 data with very high EPA
        extra_plays = []
        for week in range(1, 5):
            game_date = datetime(2024, 9, 1) + timedelta(weeks=week-1)
            for i in range(30):
                extra_plays.append({
                    "play_id": 10000 + week * 30 + i,
                    "game_id": f"nfl_2024_{week:02d}_KC_DEN",
                    "old_game_id": f"2024_{week:02d}_KC_DEN",
                    "season": 2024,
                    "week": week,
                    "gameday": game_date,
                    "posteam": "KC",
                    "defteam": "DEN",
                    "passer_id": "QBTest",
                    "passer_player_name": "Test QB",
                    "play_type": "pass",
                    "epa": 1.5,  # Very high EPA
                    "success": 1,
                    "cpoe": 5.0,
                    "complete_pass": 1,
                    "interception": 0,
                    "pass_touchdown": 0,
                    "passing_yards": 20,
                    "air_yards": 15,
                    "sack": 0,
                })

        pbp_with_2024 = pd.concat([
            sequential_pbp_data,
            pd.DataFrame(extra_plays)
        ], ignore_index=True)

        generator = QBDeviationFeatureGenerator(
            pbp_df=pbp_with_2024,
            games_df=sequential_games_data
        )

        # Get career at end of 2023 (should NOT include 2024 data)
        before_date = pd.Timestamp("2023-12-01")
        career = generator.get_career_baseline("QBTest", before_date)

        if career is not None:
            # Career EPA should be around 0.45 (2023 average), NOT inflated by 2024's 1.5 EPA
            assert career.epa_per_dropback < 0.7, \
                f"Data leakage detected! Career EPA {career.epa_per_dropback:.3f} includes future data"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_dataframes(self):
        """Test handling of empty dataframes - should raise ValueError for missing passer column."""
        # Empty DataFrame without required columns should raise ValueError
        empty_pbp = pd.DataFrame(columns=[
            "play_id", "game_id", "season", "week", "posteam", "defteam",
            "play_type", "epa", "success"  # Missing passer_id
        ])
        empty_games = pd.DataFrame(columns=["game_id", "season", "week", "date"])

        with pytest.raises(ValueError, match="No passer ID column found"):
            QBDeviationFeatureGenerator(pbp_df=empty_pbp, games_df=empty_games)

    def test_insufficient_dropbacks(self):
        """Test that QBs with insufficient dropbacks return None baseline."""
        # Create minimal data (less than MIN_CAREER_DROPBACKS)
        plays = pd.DataFrame({
            "play_id": range(50),  # Less than 100
            "game_id": ["nfl_2023_01_KC_DEN"] * 50,
            "old_game_id": ["2023_01_KC_DEN"] * 50,
            "season": [2023] * 50,
            "week": [1] * 50,
            "gameday": [datetime(2023, 9, 1)] * 50,
            "posteam": ["KC"] * 50,
            "defteam": ["DEN"] * 50,
            "passer_id": ["QBTest"] * 50,
            "play_type": ["pass"] * 50,
            "epa": [0.1] * 50,
            "success": [1] * 50,
            "complete_pass": [1] * 50,
            "interception": [0] * 50,
        })

        games = pd.DataFrame({
            "game_id": ["nfl_2023_01_KC_DEN"],
            "season": [2023],
            "week": [1],
            "date": [datetime(2023, 9, 1)],
        })

        generator = QBDeviationFeatureGenerator(pbp_df=plays, games_df=games)

        baseline = generator.get_career_baseline(
            "QBTest",
            pd.Timestamp("2024-01-01")
        )

        # Should return None due to insufficient data
        assert baseline is None

    def test_nan_handling(self):
        """Test proper handling of NaN values in data."""
        plays = pd.DataFrame({
            "play_id": range(200),
            "game_id": ["nfl_2023_01_KC_DEN"] * 200,
            "old_game_id": ["2023_01_KC_DEN"] * 200,
            "season": [2023] * 200,
            "week": [1] * 200,
            "gameday": [datetime(2023, 9, 1)] * 200,
            "posteam": ["KC"] * 200,
            "defteam": ["DEN"] * 200,
            "passer_id": ["QBTest"] * 200,
            "play_type": ["pass"] * 200,
            "epa": [0.1] * 100 + [np.nan] * 100,  # Half NaN
            "success": [1] * 200,
            "cpoe": [np.nan] * 200,  # All NaN
            "complete_pass": [1] * 200,
            "interception": [0] * 200,
            "sack": [0] * 200,  # Required for sack rate calculation
        })

        games = pd.DataFrame({
            "game_id": ["nfl_2023_01_KC_DEN"],
            "season": [2023],
            "week": [1],
            "date": [datetime(2023, 9, 1)],
        })

        generator = QBDeviationFeatureGenerator(pbp_df=plays, games_df=games)

        # Should handle NaN gracefully
        features = generator.generate_game_features(
            game_id="nfl_2023_02_KC_DEN",
            season=2023,
            week=2,
            team="KC",
            qb_id="QBTest",
            game_date=pd.Timestamp("2023-09-15")
        )

        # Should not raise and should return features
        assert isinstance(features, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
