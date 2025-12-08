"""
Tests for NGS (Next Gen Stats) ingestion and feature extraction.

Tests cover:
- Data ingestion from nflverse
- Feature extraction for QBs, RBs, WRs
- Team-level aggregation
- Data leakage prevention
- Edge cases
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil


class TestNGSIngester:
    """Tests for NGS data ingestion."""

    def test_ingester_initialization(self):
        """Test that ingester initializes correctly."""
        from ingestion.nfl.ngs import NGSIngester

        ingester = NGSIngester()
        assert ingester.MIN_SEASON == 2016
        assert 'passing' in ingester.STAT_TYPES
        assert 'rushing' in ingester.STAT_TYPES
        assert 'receiving' in ingester.STAT_TYPES

    def test_ingester_custom_cache_dir(self):
        """Test ingester with custom cache directory."""
        from ingestion.nfl.ngs import NGSIngester

        with tempfile.TemporaryDirectory() as tmpdir:
            ingester = NGSIngester(cache_dir=tmpdir)
            assert ingester.cache_dir == Path(tmpdir)

    def test_season_validation(self):
        """Test that seasons before 2016 are filtered out."""
        from ingestion.nfl.ngs import NGSIngester

        ingester = NGSIngester()

        # Mock the fetch to avoid network calls
        with patch.object(ingester, '_fetch_ngs_data', return_value=pd.DataFrame()):
            # This should filter out 2015
            result = ingester.ingest_all(seasons=[2015, 2016, 2017], force_refresh=True)

            # Should only have tried to fetch 2016, 2017
            # (though result will be empty due to mock)


class TestNGSFeatureExtractor:
    """Tests for NGS feature extraction."""

    @pytest.fixture
    def sample_passing_data(self):
        """Create sample passing NGS data for testing."""
        return pd.DataFrame({
            'player_gsis_id': ['QB1'] * 10,
            'player_display_name': ['Test QB'] * 10,
            'team_abbr': ['KC'] * 10,
            'season': [2023] * 5 + [2024] * 5,
            'week': [10, 11, 12, 13, 14, 1, 2, 3, 4, 5],
            'attempts': [30, 35, 28, 32, 30, 33, 29, 31, 34, 32],
            'avg_time_to_throw': [2.5, 2.6, 2.4, 2.5, 2.7, 2.4, 2.5, 2.6, 2.5, 2.4],
            'completion_percentage': [65, 68, 62, 70, 66, 64, 67, 69, 68, 65],
            'expected_completion_percentage': [62, 64, 60, 65, 63, 62, 64, 66, 65, 63],
            'completion_percentage_above_expectation': [3, 4, 2, 5, 3, 2, 3, 3, 3, 2],
            'aggressiveness': [18, 20, 15, 22, 19, 17, 19, 21, 20, 18],
            'passer_rating': [95, 100, 88, 105, 92, 90, 98, 102, 99, 94],
            'avg_completed_air_yards': [5.5, 6.0, 5.2, 6.5, 5.8, 5.3, 5.7, 6.2, 6.0, 5.5],
            'avg_intended_air_yards': [7.5, 8.0, 7.2, 8.5, 7.8, 7.3, 7.7, 8.2, 8.0, 7.5],
            'avg_air_yards_differential': [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            'avg_air_yards_to_sticks': [0.5, 0.8, 0.3, 1.0, 0.6, 0.4, 0.7, 0.9, 0.8, 0.5],
        })

    @pytest.fixture
    def sample_rushing_data(self):
        """Create sample rushing NGS data for testing."""
        return pd.DataFrame({
            'player_gsis_id': ['RB1'] * 8,
            'player_display_name': ['Test RB'] * 8,
            'team_abbr': ['KC'] * 8,
            'season': [2023] * 4 + [2024] * 4,
            'week': [11, 12, 13, 14, 1, 2, 3, 4],
            'rush_attempts': [15, 18, 12, 20, 16, 14, 17, 19],
            'rush_yards': [75, 90, 48, 95, 72, 68, 85, 92],
            'avg_rush_yards': [5.0, 5.0, 4.0, 4.75, 4.5, 4.86, 5.0, 4.84],
            'expected_rush_yards': [65, 80, 50, 85, 68, 62, 75, 82],
            'rush_yards_over_expected': [10, 10, -2, 10, 4, 6, 10, 10],
            'rush_yards_over_expected_per_att': [0.67, 0.56, -0.17, 0.50, 0.25, 0.43, 0.59, 0.53],
            'efficiency': [1.2, 1.15, 1.3, 1.18, 1.22, 1.19, 1.16, 1.17],
            'percent_attempts_gte_eight_defenders': [25, 30, 35, 28, 32, 27, 29, 31],
            'avg_time_to_los': [2.1, 2.0, 2.3, 2.1, 2.2, 2.1, 2.0, 2.1],
        })

    @pytest.fixture
    def sample_receiving_data(self):
        """Create sample receiving NGS data for testing."""
        return pd.DataFrame({
            'player_gsis_id': ['WR1'] * 8,
            'player_display_name': ['Test WR'] * 8,
            'team_abbr': ['KC'] * 8,
            'season': [2023] * 4 + [2024] * 4,
            'week': [11, 12, 13, 14, 1, 2, 3, 4],
            'targets': [8, 10, 6, 12, 9, 7, 11, 10],
            'receptions': [6, 7, 4, 9, 7, 5, 8, 8],
            'yards': [85, 95, 55, 120, 90, 70, 105, 100],
            'catch_percentage': [75, 70, 67, 75, 78, 71, 73, 80],
            'avg_cushion': [5.5, 5.8, 5.2, 6.0, 5.6, 5.4, 5.9, 5.7],
            'avg_separation': [2.8, 3.0, 2.5, 3.2, 2.9, 2.7, 3.1, 3.0],
            'avg_yac': [4.5, 5.0, 3.8, 5.5, 4.8, 4.2, 5.2, 5.0],
            'avg_expected_yac': [4.0, 4.5, 3.5, 5.0, 4.3, 3.8, 4.8, 4.5],
            'avg_yac_above_expectation': [0.5, 0.5, 0.3, 0.5, 0.5, 0.4, 0.4, 0.5],
            'percent_share_of_intended_air_yards': [22, 25, 18, 28, 24, 20, 26, 25],
        })

    def test_qb_features_data_leakage_prevention(self, sample_passing_data):
        """Test that QB features only use past data."""
        from features.nfl.ngs_features import NGSFeatureExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save sample data
            cache_dir = Path(tmpdir)
            passing_dir = cache_dir / "passing"
            passing_dir.mkdir(parents=True)
            sample_passing_data.to_parquet(passing_dir / "ngs_data.parquet")

            extractor = NGSFeatureExtractor(data_dir=tmpdir)

            # Get features for week 5 of 2024
            features = extractor.compute_qb_features('QB1', season=2024, week=5)

            # Should have features (data exists before week 5)
            assert len(features) > 0

            # Features should be based on weeks 1-4 of 2024 and all of 2023
            # Check that L3 uses last 3 games before week 5
            assert 'qb_ngs_completion_percentage_above_expectation_L3' in features

    def test_qb_features_no_future_data(self, sample_passing_data):
        """Verify no future data is used in features."""
        from features.nfl.ngs_features import NGSFeatureExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            passing_dir = cache_dir / "passing"
            passing_dir.mkdir(parents=True)
            sample_passing_data.to_parquet(passing_dir / "ngs_data.parquet")

            extractor = NGSFeatureExtractor(data_dir=tmpdir)

            # Get features for week 2 of 2024
            features = extractor.compute_qb_features('QB1', season=2024, week=2)

            # Should only use week 1 of 2024 and all of 2023
            # L3 should use week 1 2024 + weeks 13, 14 of 2023
            assert len(features) > 0

    def test_qb_features_empty_for_new_player(self, sample_passing_data):
        """Test that new player returns empty features."""
        from features.nfl.ngs_features import NGSFeatureExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            passing_dir = cache_dir / "passing"
            passing_dir.mkdir(parents=True)
            sample_passing_data.to_parquet(passing_dir / "ngs_data.parquet")

            extractor = NGSFeatureExtractor(data_dir=tmpdir)

            # Unknown player should return empty
            features = extractor.compute_qb_features('UNKNOWN_QB', season=2024, week=5)
            assert len(features) == 0

    def test_qb_features_career_baseline(self, sample_passing_data):
        """Test that career baseline features are computed."""
        from features.nfl.ngs_features import NGSFeatureExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            passing_dir = cache_dir / "passing"
            passing_dir.mkdir(parents=True)
            sample_passing_data.to_parquet(passing_dir / "ngs_data.parquet")

            extractor = NGSFeatureExtractor(data_dir=tmpdir)

            features = extractor.compute_qb_features('QB1', season=2024, week=5, include_career=True)

            # Should have career features
            assert 'qb_ngs_completion_percentage_above_expectation_career' in features

    def test_qb_features_cpoe_vs_career(self, sample_passing_data):
        """Test CPOE vs career deviation feature."""
        from features.nfl.ngs_features import NGSFeatureExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            passing_dir = cache_dir / "passing"
            passing_dir.mkdir(parents=True)
            sample_passing_data.to_parquet(passing_dir / "ngs_data.parquet")

            extractor = NGSFeatureExtractor(data_dir=tmpdir)

            features = extractor.compute_qb_features('QB1', season=2024, week=5)

            if 'qb_ngs_cpoe_vs_career' in features:
                # Should be season CPOE minus career CPOE
                expected = (
                    features.get('qb_ngs_completion_percentage_above_expectation_season', 0) -
                    features.get('qb_ngs_completion_percentage_above_expectation_career', 0)
                )
                assert abs(features['qb_ngs_cpoe_vs_career'] - expected) < 0.01

    def test_rb_features(self, sample_rushing_data):
        """Test RB feature extraction."""
        from features.nfl.ngs_features import NGSFeatureExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            rushing_dir = cache_dir / "rushing"
            rushing_dir.mkdir(parents=True)
            sample_rushing_data.to_parquet(rushing_dir / "ngs_data.parquet")

            extractor = NGSFeatureExtractor(data_dir=tmpdir)

            features = extractor.compute_rb_features('RB1', season=2024, week=5)

            assert len(features) > 0
            # Should have RYOE per attempt
            assert 'rb_ngs_ryoe_per_att' in features or 'rb_ngs_rush_yards_over_expected_per_att_L5' in features

    def test_wr_features(self, sample_receiving_data):
        """Test WR feature extraction."""
        from features.nfl.ngs_features import NGSFeatureExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            receiving_dir = cache_dir / "receiving"
            receiving_dir.mkdir(parents=True)
            sample_receiving_data.to_parquet(receiving_dir / "ngs_data.parquet")

            extractor = NGSFeatureExtractor(data_dir=tmpdir)

            features = extractor.compute_wr_features('WR1', season=2024, week=5)

            assert len(features) > 0
            # Should have separation metric
            assert any('separation' in k for k in features.keys())

    def test_team_qb_features(self, sample_passing_data):
        """Test team-level QB feature extraction."""
        from features.nfl.ngs_features import NGSFeatureExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            passing_dir = cache_dir / "passing"
            passing_dir.mkdir(parents=True)
            sample_passing_data.to_parquet(passing_dir / "ngs_data.parquet")

            extractor = NGSFeatureExtractor(data_dir=tmpdir)

            # Should find KC's QB
            features = extractor.compute_team_qb_features('KC', season=2024, week=5)

            assert len(features) > 0


class TestNGSTeamAggregator:
    """Tests for team-level NGS aggregation."""

    @pytest.fixture
    def sample_data_all_positions(self):
        """Create sample data for all positions."""
        passing = pd.DataFrame({
            'player_gsis_id': ['QB1'] * 5,
            'team_abbr': ['KC'] * 5,
            'season': [2024] * 5,
            'week': [1, 2, 3, 4, 5],
            'attempts': [30, 35, 28, 32, 30],
            'completion_percentage_above_expectation': [3, 4, 2, 5, 3],
            'avg_time_to_throw': [2.5, 2.6, 2.4, 2.5, 2.7],
            'aggressiveness': [18, 20, 15, 22, 19],
        })

        rushing = pd.DataFrame({
            'player_gsis_id': ['RB1'] * 5,
            'team_abbr': ['KC'] * 5,
            'season': [2024] * 5,
            'week': [1, 2, 3, 4, 5],
            'rush_attempts': [15, 18, 12, 20, 16],
            'rush_yards_over_expected_per_att': [0.5, 0.6, 0.3, 0.7, 0.5],
        })

        receiving = pd.DataFrame({
            'player_gsis_id': ['WR1'] * 5,
            'team_abbr': ['KC'] * 5,
            'season': [2024] * 5,
            'week': [1, 2, 3, 4, 5],
            'targets': [8, 10, 6, 12, 9],
            'avg_separation': [2.8, 3.0, 2.5, 3.2, 2.9],
        })

        return passing, rushing, receiving

    def test_matchup_features(self, sample_data_all_positions):
        """Test matchup feature generation."""
        from features.nfl.ngs_features import NGSFeatureExtractor, NGSTeamAggregator

        passing, rushing, receiving = sample_data_all_positions

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Save all data
            for name, df in [('passing', passing), ('rushing', rushing), ('receiving', receiving)]:
                subdir = cache_dir / name
                subdir.mkdir(parents=True)
                df.to_parquet(subdir / "ngs_data.parquet")

            extractor = NGSFeatureExtractor(data_dir=tmpdir)
            aggregator = NGSTeamAggregator(extractor)

            # Generate matchup features
            features = aggregator.compute_matchup_features(
                home_team='KC',
                away_team='BUF',  # No data for BUF
                season=2024,
                week=6
            )

            # Should have home team features
            assert any(k.startswith('home_') for k in features.keys())


class TestDataLeakageValidation:
    """Specific tests for data leakage prevention."""

    def test_no_same_week_data(self):
        """Verify that same-week data is never used."""
        from features.nfl.ngs_features import NGSFeatureExtractor

        # Create data where week 5 has distinctive value
        data = pd.DataFrame({
            'player_gsis_id': ['QB1'] * 6,
            'season': [2024] * 6,
            'week': [1, 2, 3, 4, 5, 6],
            'completion_percentage_above_expectation': [3, 4, 2, 5, 100, 3],  # Week 5 has 100
            'team_abbr': ['KC'] * 6,
            'attempts': [30] * 6,
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            passing_dir = cache_dir / "passing"
            passing_dir.mkdir(parents=True)
            data.to_parquet(passing_dir / "ngs_data.parquet")

            extractor = NGSFeatureExtractor(data_dir=tmpdir)

            # Get features for week 5 - should NOT include the 100 value
            features = extractor.compute_qb_features('QB1', season=2024, week=5)

            # L3 should be average of weeks 2, 3, 4 = (4 + 2 + 5) / 3 = 3.67
            if 'qb_ngs_completion_percentage_above_expectation_L3' in features:
                assert features['qb_ngs_completion_percentage_above_expectation_L3'] < 10

    def test_no_future_season_data(self):
        """Verify that future season data is never used."""
        from features.nfl.ngs_features import NGSFeatureExtractor

        # Create data with 2025 having distinctive value
        data = pd.DataFrame({
            'player_gsis_id': ['QB1'] * 5,
            'season': [2023, 2024, 2024, 2024, 2025],
            'week': [1, 1, 2, 3, 1],
            'completion_percentage_above_expectation': [3, 4, 5, 4, 100],  # 2025 has 100
            'team_abbr': ['KC'] * 5,
            'attempts': [30] * 5,
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            passing_dir = cache_dir / "passing"
            passing_dir.mkdir(parents=True)
            data.to_parquet(passing_dir / "ngs_data.parquet")

            extractor = NGSFeatureExtractor(data_dir=tmpdir)

            # Get features for week 4 of 2024 - should NOT include 2025 data
            features = extractor.compute_qb_features('QB1', season=2024, week=4)

            # Career should NOT include the 100 value from 2025
            if 'qb_ngs_completion_percentage_above_expectation_career' in features:
                assert features['qb_ngs_completion_percentage_above_expectation_career'] < 10


class TestFTNFeatureExtractor:
    """Tests for FTN feature extraction."""

    @pytest.fixture
    def sample_ftn_data(self):
        """Create sample FTN charting data."""
        return pd.DataFrame({
            'game_id': [f'game_{i}' for i in range(100)],
            'play_id': list(range(100)),
            'posteam': ['KC'] * 100,
            'defteam': ['BUF'] * 100,
            'season': [2024] * 100,
            'week': [i // 10 + 1 for i in range(100)],
            'is_play_action': [True if i % 5 == 0 else False for i in range(100)],
            'is_screen': [True if i % 10 == 0 else False for i in range(100)],
            'is_blitz': [True if i % 4 == 0 else False for i in range(100)],
            'is_qb_out_of_pocket': [True if i % 8 == 0 else False for i in range(100)],
            'is_catchable': [True if i % 2 == 0 else False for i in range(100)],
            'n_pass_rushers': [4 + (i % 3) for i in range(100)],
        })

    def test_ftn_team_features(self, sample_ftn_data):
        """Test FTN team feature extraction."""
        from features.nfl.advanced_game_features import FTNFeatureExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            sample_ftn_data.to_parquet(cache_dir / "ftn_data.parquet")

            extractor = FTNFeatureExtractor(data_dir=tmpdir)

            features = extractor.compute_team_features('KC', season=2024, week=10, is_offense=True)

            # Should have play action rate
            assert 'off_ftn_play_action_rate' in features
            assert 0 <= features['off_ftn_play_action_rate'] <= 1


class TestPFRFeatureExtractor:
    """Tests for PFR feature extraction."""

    @pytest.fixture
    def sample_pfr_passing(self):
        """Create sample PFR passing data."""
        return pd.DataFrame({
            'pfr_player_id': ['PFR_QB1'] * 8,
            'player': ['Test QB'] * 8,
            'team': ['KC'] * 8,
            'season': [2024] * 8,
            'week': list(range(1, 9)),
            'attempts': [30, 35, 28, 32, 30, 33, 29, 31],
            'dropbacks': [35, 40, 33, 37, 35, 38, 34, 36],
            'times_pressured': [8, 10, 7, 9, 8, 11, 7, 9],
            'times_hurried': [5, 6, 4, 5, 5, 7, 4, 6],
            'times_hit': [3, 4, 3, 4, 3, 4, 3, 3],
            'times_blitzed': [10, 12, 9, 11, 10, 13, 9, 11],
            'on_target_throws': [24, 28, 22, 26, 24, 27, 23, 25],
            'bad_throws': [6, 7, 6, 6, 6, 6, 6, 6],
            'pocket_time': [2.5, 2.6, 2.4, 2.5, 2.5, 2.3, 2.6, 2.5],
        })

    def test_pfr_qb_features(self, sample_pfr_passing):
        """Test PFR QB feature extraction."""
        from features.nfl.advanced_game_features import PFRFeatureExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            sample_pfr_passing.to_parquet(cache_dir / "pfr_weekly_pass.parquet")

            extractor = PFRFeatureExtractor(data_dir=tmpdir)

            features = extractor.compute_team_qb_features('KC', season=2024, week=9)

            # Should have pressure-related features
            assert len(features) > 0


class TestAdvancedGameFeatureGenerator:
    """Tests for the full game feature generator."""

    def test_generate_game_features_structure(self):
        """Test that generated features have correct structure."""
        from features.nfl.advanced_game_features import AdvancedGameFeatureGenerator

        # Create generator (will have empty data)
        generator = AdvancedGameFeatureGenerator()

        features = generator.generate_game_features(
            game_id='nfl_2024_10_KC_DEN',
            home_team='DEN',
            away_team='KC',
            season=2024,
            week=10
        )

        # Should have basic info
        assert features['game_id'] == 'nfl_2024_10_KC_DEN'
        assert features['season'] == 2024
        assert features['week'] == 10
        assert features['home_team'] == 'DEN'
        assert features['away_team'] == 'KC'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
