"""
Metric regression tests for Phase 1 baseline.

These tests ensure that:
1. Pipeline metrics don't regress beyond tolerance windows
2. Configuration matches the frozen baseline
3. Sanity checks are always satisfied
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml


class TestConfigSnapshot:
    """Tests to verify configuration matches frozen baseline."""

    @pytest.fixture
    def baseline_config(self):
        config_path = Path(__file__).parent.parent / "config" / "nfl_phase1_baseline.yml"
        with open(config_path) as f:
            return yaml.safe_load(f)

    @pytest.fixture
    def model_config(self):
        config_path = Path(__file__).parent.parent / "config" / "models" / "nfl_baseline.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)

    @pytest.fixture
    def backtest_config(self):
        config_path = Path(__file__).parent.parent / "config" / "evaluation" / "backtest_config.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_logistic_regression_config_matches(self, baseline_config, model_config):
        """Test that logistic regression config matches baseline."""
        baseline = baseline_config["models"]["logistic_regression"]
        actual = model_config["models"]["logistic_regression"]

        assert actual["C"] == baseline["C"], f"C mismatch: expected {baseline['C']}, got {actual['C']}"
        assert actual["max_iter"] == baseline["max_iter"]
        assert actual["random_state"] == baseline["random_state"]

    def test_gbm_config_matches(self, baseline_config, model_config):
        """Test that GBM config matches baseline."""
        baseline = baseline_config["models"]["gradient_boosting"]
        actual = model_config["models"]["gradient_boosting"]

        assert actual["n_estimators"] == baseline["n_estimators"]
        assert actual["max_depth"] == baseline["max_depth"]
        assert actual["learning_rate"] == baseline["learning_rate"]
        assert actual["random_state"] == baseline["random_state"]

    def test_splits_config_matches(self, baseline_config, backtest_config):
        """Test that train/test splits match baseline."""
        baseline_splits = baseline_config["splits"]
        actual_splits = backtest_config["splits"]

        assert actual_splits["train_seasons"] == baseline_splits["train_seasons"]
        assert actual_splits["validation_season"] == baseline_splits["validation_season"]
        assert actual_splits["test_season"] == baseline_splits["test_season"]

    def test_edge_thresholds_match(self, baseline_config, backtest_config):
        """Test that edge thresholds match baseline."""
        baseline = baseline_config["evaluation"]["edge_thresholds"]
        actual = backtest_config["roi"]["edge_thresholds"]

        assert actual == baseline, f"Edge thresholds mismatch: expected {baseline}, got {actual}"


class TestSampleMetricRegression:
    """Tests to ensure sample pipeline metrics don't regress."""

    @pytest.fixture
    def metrics_snapshot(self):
        snapshot_path = (
            Path(__file__).parent.parent
            / "config"
            / "snapshots"
            / "phase1_sample_metrics.yaml"
        )
        with open(snapshot_path) as f:
            return yaml.safe_load(f)

    @pytest.fixture
    def pipeline_metrics(self):
        """Run sample pipeline and get metrics."""
        from orchestration.pipelines.phase1_sample_pipeline import run_sample_pipeline
        results = run_sample_pipeline()
        return results["metrics"]

    def test_accuracy_in_range(self, pipeline_metrics, metrics_snapshot):
        """Test that accuracy is within expected range."""
        actual = pipeline_metrics["accuracy"]
        sanity = metrics_snapshot["sanity_checks"]["accuracy"]

        # Sanity check (must always pass)
        assert sanity["min"] <= actual <= sanity["max"], (
            f"Accuracy {actual:.4f} outside sanity range [{sanity['min']}, {sanity['max']}]"
        )

        # Expected range check
        expected = metrics_snapshot["sample_baseline"]["accuracy"]
        assert expected["min"] <= actual <= expected["max"], (
            f"Accuracy {actual:.4f} outside expected range [{expected['min']}, {expected['max']}]"
        )

    def test_brier_score_in_range(self, pipeline_metrics, metrics_snapshot):
        """Test that Brier score is within expected range."""
        actual = pipeline_metrics["brier_score"]
        sanity = metrics_snapshot["sanity_checks"]["brier_score"]

        # Sanity check
        assert sanity["min"] <= actual <= sanity["max"], (
            f"Brier score {actual:.4f} outside sanity range [{sanity['min']}, {sanity['max']}]"
        )

        # Expected range check
        expected = metrics_snapshot["sample_baseline"]["brier_score"]
        assert expected["min"] <= actual <= expected["max"], (
            f"Brier score {actual:.4f} outside expected range [{expected['min']}, {expected['max']}]"
        )

    def test_log_loss_in_range(self, pipeline_metrics, metrics_snapshot):
        """Test that log loss is within expected range."""
        actual = pipeline_metrics["log_loss"]
        sanity = metrics_snapshot["sanity_checks"]["log_loss"]

        # Sanity check
        assert sanity["min"] <= actual <= sanity["max"], (
            f"Log loss {actual:.4f} outside sanity range [{sanity['min']}, {sanity['max']}]"
        )

        # Expected range check
        expected = metrics_snapshot["sample_baseline"]["log_loss"]
        assert expected["min"] <= actual <= expected["max"], (
            f"Log loss {actual:.4f} outside expected range [{expected['min']}, {expected['max']}]"
        )

    def test_sufficient_training_data(self, pipeline_metrics, metrics_snapshot):
        """Test that we have sufficient training data."""
        n_train = pipeline_metrics["n_train"]
        min_required = metrics_snapshot["sanity_checks"]["n_train"]["min"]

        assert n_train >= min_required, (
            f"Training data too small: {n_train} < {min_required}"
        )

    def test_sufficient_test_data(self, pipeline_metrics, metrics_snapshot):
        """Test that we have sufficient test data."""
        n_test = pipeline_metrics["n_test"]
        min_required = metrics_snapshot["sanity_checks"]["n_test"]["min"]

        assert n_test >= min_required, (
            f"Test data too small: {n_test} < {min_required}"
        )


class TestNoNaNValues:
    """Tests to ensure no NaN values in metrics."""

    @pytest.fixture
    def pipeline_metrics(self):
        from orchestration.pipelines.phase1_sample_pipeline import run_sample_pipeline
        results = run_sample_pipeline()
        return results["metrics"]

    def test_accuracy_not_nan(self, pipeline_metrics):
        """Test that accuracy is not NaN."""
        import math
        assert not math.isnan(pipeline_metrics["accuracy"]), "Accuracy is NaN"

    def test_brier_score_not_nan(self, pipeline_metrics):
        """Test that Brier score is not NaN."""
        import math
        assert not math.isnan(pipeline_metrics["brier_score"]), "Brier score is NaN"

    def test_log_loss_not_nan(self, pipeline_metrics):
        """Test that log loss is not NaN."""
        import math
        assert not math.isnan(pipeline_metrics["log_loss"]), "Log loss is NaN"
