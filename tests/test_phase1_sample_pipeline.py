"""
Integration tests for the sample Phase 1 pipeline.

These tests verify:
1. Sample data loads correctly
2. Pipeline produces expected artifacts
3. Metrics are finite and within sanity ranges
4. No data leakage in rolling features
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd


class TestSampleDataIntegrity:
    """Tests for sample data integrity."""

    @pytest.fixture
    def sample_data_dir(self):
        return Path(__file__).parent.parent / "data" / "nfl" / "sample" / "staged"

    def test_games_csv_exists(self, sample_data_dir):
        """Test that games.csv exists."""
        assert (sample_data_dir / "games.csv").exists()

    def test_markets_csv_exists(self, sample_data_dir):
        """Test that markets.csv exists."""
        assert (sample_data_dir / "markets.csv").exists()

    def test_team_stats_csv_exists(self, sample_data_dir):
        """Test that team_stats.csv exists."""
        assert (sample_data_dir / "team_stats.csv").exists()

    def test_games_data_valid(self, sample_data_dir):
        """Test that games data has required columns and valid values."""
        df = pd.read_csv(sample_data_dir / "games.csv")

        required_cols = ["game_id", "season", "week", "date", "home_team", "away_team", "home_score", "away_score"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

        # Check no null values in required columns
        assert df[required_cols].notna().all().all(), "Found null values in required columns"

        # Check scores are reasonable
        assert (df["home_score"] >= 0).all(), "Found negative home scores"
        assert (df["away_score"] >= 0).all(), "Found negative away scores"
        assert (df["home_score"] <= 70).all(), "Found unreasonably high home scores"
        assert (df["away_score"] <= 70).all(), "Found unreasonably high away scores"

    def test_markets_data_valid(self, sample_data_dir):
        """Test that markets data has required columns and valid values."""
        df = pd.read_csv(sample_data_dir / "markets.csv")

        required_cols = ["game_id", "season", "week", "close_spread", "close_total"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

        # Check spreads are reasonable (-30 to +30)
        assert (df["close_spread"] >= -30).all(), "Found unreasonably large negative spread"
        assert (df["close_spread"] <= 30).all(), "Found unreasonably large positive spread"

        # Check totals are reasonable (20 to 70)
        assert (df["close_total"] >= 20).all(), "Found unreasonably low total"
        assert (df["close_total"] <= 70).all(), "Found unreasonably high total"

    def test_team_stats_data_valid(self, sample_data_dir):
        """Test that team stats data has required columns and valid values."""
        df = pd.read_csv(sample_data_dir / "team_stats.csv")

        required_cols = ["game_id", "team", "is_home", "points_for", "points_against"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

        # Each game should have exactly 2 team stats rows (home and away)
        games_per_stat = df.groupby("game_id").size()
        assert (games_per_stat == 2).all(), "Each game should have exactly 2 team stat entries"

    def test_game_ids_consistent(self, sample_data_dir):
        """Test that game IDs are consistent across all files."""
        games_df = pd.read_csv(sample_data_dir / "games.csv")
        markets_df = pd.read_csv(sample_data_dir / "markets.csv")
        team_stats_df = pd.read_csv(sample_data_dir / "team_stats.csv")

        games_ids = set(games_df["game_id"])
        markets_ids = set(markets_df["game_id"])
        team_stats_ids = set(team_stats_df["game_id"])

        # Markets should match games
        assert markets_ids == games_ids, "Markets game IDs don't match games"

        # Team stats should match games
        assert team_stats_ids == games_ids, "Team stats game IDs don't match games"


class TestSamplePipelineExecution:
    """Tests for sample pipeline execution."""

    @pytest.fixture
    def run_pipeline(self):
        """Run the sample pipeline and return results."""
        from orchestration.pipelines.phase1_sample_pipeline import run_sample_pipeline
        return run_sample_pipeline()

    def test_pipeline_completes(self, run_pipeline):
        """Test that pipeline runs without error."""
        assert run_pipeline is not None

    def test_pipeline_produces_artifacts(self, run_pipeline):
        """Test that pipeline produces expected output files."""
        artifacts = run_pipeline["artifacts"]

        assert Path(artifacts["game_features"]).exists(), "Game features file not created"
        assert Path(artifacts["team_features"]).exists(), "Team features file not created"
        assert Path(artifacts["report"]).exists(), "Report file not created"

    def test_game_features_valid(self, run_pipeline):
        """Test that game features have valid structure."""
        game_features_path = Path(run_pipeline["artifacts"]["game_features"])
        df = pd.read_parquet(game_features_path)

        # Check required columns
        required = ["game_id", "season", "week", "home_team", "away_team", "home_score", "away_score"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

        # Check feature columns exist
        feature_cols = [col for col in df.columns if col.startswith(("home_", "away_")) and "score" not in col]
        assert len(feature_cols) > 0, "No feature columns found"

        # Check no NaN in game identifiers
        assert df["game_id"].notna().all(), "Found null game_ids"

    def test_metrics_valid(self, run_pipeline):
        """Test that metrics are finite and within sanity ranges."""
        metrics = run_pipeline["metrics"]

        # Check accuracy is valid
        assert 0 <= metrics["accuracy"] <= 1, f"Invalid accuracy: {metrics['accuracy']}"
        assert pd.notna(metrics["accuracy"]), "Accuracy is NaN"

        # Check Brier score is valid (0 to 1, lower is better)
        assert 0 <= metrics["brier_score"] <= 1, f"Invalid Brier score: {metrics['brier_score']}"
        assert pd.notna(metrics["brier_score"]), "Brier score is NaN"

        # Check log loss is valid (>= 0, lower is better)
        assert metrics["log_loss"] >= 0, f"Invalid log loss: {metrics['log_loss']}"
        assert pd.notna(metrics["log_loss"]), "Log loss is NaN"


class TestNoLeakage:
    """Tests to ensure no data leakage in feature engineering."""

    @pytest.fixture
    def sample_data_dir(self):
        return Path(__file__).parent.parent / "data" / "nfl" / "sample" / "staged"

    def test_rolling_features_exclude_current_game(self, sample_data_dir):
        """
        Test that rolling features exclude the current game.

        For a team's first game, rolling features should be 0 (no history).
        For subsequent games, features should only use prior games.
        """
        from orchestration.pipelines.phase1_sample_pipeline import (
            load_sample_data,
            calculate_rolling_features,
        )

        games_df, _, team_stats_df = load_sample_data()
        features_df = calculate_rolling_features(team_stats_df, games_df)

        # Check first game for each team has 0 rolling features
        for team in features_df["team"].unique():
            team_features = features_df[features_df["team"] == team].sort_values("date")
            first_game = team_features.iloc[0]

            # First game should have 0 for all rolling features
            rolling_cols = [col for col in first_game.index if "last4" in col or "last8" in col or "last16" in col]
            for col in rolling_cols:
                assert first_game[col] == 0.0, f"Team {team} first game has non-zero {col}: {first_game[col]}"

    def test_features_use_only_past_data(self, sample_data_dir):
        """
        Test that features for game N only use data from games 1 to N-1.

        This verifies the rolling window calculation is correct.
        """
        from orchestration.pipelines.phase1_sample_pipeline import (
            load_sample_data,
            calculate_rolling_features,
        )

        games_df, _, team_stats_df = load_sample_data()
        features_df = calculate_rolling_features(team_stats_df, games_df)

        # For a team with at least 5 games, verify win_rate_last4
        for team in features_df["team"].unique():
            team_df = features_df[features_df["team"] == team].sort_values("date").reset_index(drop=True)

            if len(team_df) < 5:
                continue

            # Check game 5 (index 4) - should use games 1-4
            game_5 = team_df.iloc[4]
            games_1_to_4 = team_df.iloc[:4]

            # Calculate expected win rate
            expected_win_rate = games_1_to_4["win"].mean()
            actual_win_rate = game_5["win_rate_last4"]

            assert abs(expected_win_rate - actual_win_rate) < 0.001, (
                f"Team {team} game 5 win_rate_last4 mismatch: "
                f"expected {expected_win_rate:.4f}, got {actual_win_rate:.4f}"
            )


class TestReportGeneration:
    """Tests for report generation."""

    @pytest.fixture
    def report_path(self):
        return Path(__file__).parent.parent / "docs" / "reports" / "sample_phase1.md"

    def test_report_created(self, report_path):
        """Test that report file exists after pipeline run."""
        from orchestration.pipelines.phase1_sample_pipeline import run_sample_pipeline
        run_sample_pipeline()

        assert report_path.exists(), "Report file not created"

    def test_report_contains_metrics(self, report_path):
        """Test that report contains expected metrics."""
        from orchestration.pipelines.phase1_sample_pipeline import run_sample_pipeline
        run_sample_pipeline()

        with open(report_path) as f:
            content = f.read()

        assert "Accuracy" in content, "Report missing accuracy"
        assert "Brier Score" in content, "Report missing Brier score"
        assert "Log Loss" in content, "Report missing log loss"
