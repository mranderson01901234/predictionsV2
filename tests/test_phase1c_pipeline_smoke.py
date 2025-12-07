"""
Smoke test for Phase 1C pipeline.

This test verifies that:
1. Pipeline can be imported and initialized
2. Config files are valid
3. Key components can be loaded
4. Pipeline produces expected outputs when run (if data available)
"""

import sys
from pathlib import Path
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPhase1CImports:
    """Test that Phase 1C pipeline components can be imported."""

    def test_pipeline_imports(self):
        """Test that pipeline module can be imported."""
        from orchestration.pipelines.phase1c_pipeline import run_phase1c
        assert callable(run_phase1c)

    def test_trainer_imports(self):
        """Test that trainer components can be imported."""
        from models.training.trainer import (
            run_training_pipeline,
            load_backtest_config,
            load_config,
        )
        assert callable(run_training_pipeline)
        assert callable(load_backtest_config)
        assert callable(load_config)

    def test_eval_imports(self):
        """Test that evaluation components can be imported."""
        from eval.backtest import run_backtest
        from eval.reports import generate_report
        assert callable(run_backtest)
        assert callable(generate_report)

    def test_model_imports(self):
        """Test that model architectures can be imported."""
        from models.architectures.logistic_regression import LogisticRegressionModel
        from models.architectures.gradient_boosting import GradientBoostingModel
        from models.architectures.ensemble import EnsembleModel
        from models.architectures.market_baseline import MarketBaselineModel


class TestPhase1CConfig:
    """Test that Phase 1C configuration files are valid."""

    @pytest.fixture
    def project_root(self):
        return Path(__file__).parent.parent

    def test_backtest_config_exists_and_valid(self, project_root):
        """Test that backtest config exists and is valid YAML."""
        config_path = project_root / "config" / "evaluation" / "backtest_config.yaml"
        assert config_path.exists(), f"Config file not found: {config_path}"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Verify required keys
        assert "splits" in config, "Missing 'splits' key"
        assert "train_seasons" in config["splits"], "Missing 'train_seasons'"
        assert "validation_season" in config["splits"], "Missing 'validation_season'"
        assert "test_season" in config["splits"], "Missing 'test_season'"
        assert "roi" in config, "Missing 'roi' key"
        assert "edge_thresholds" in config["roi"], "Missing 'edge_thresholds'"

    def test_model_config_exists_and_valid(self, project_root):
        """Test that model config exists and is valid YAML."""
        config_path = project_root / "config" / "models" / "nfl_baseline.yaml"
        assert config_path.exists(), f"Config file not found: {config_path}"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Verify config has expected structure
        assert config is not None, "Config is empty"


class TestPhase1CPipelineSmoke:
    """Smoke test for Phase 1C pipeline execution."""

    @pytest.fixture
    def project_root(self):
        return Path(__file__).parent.parent

    def test_config_can_be_loaded(self, project_root):
        """Test that backtest config can be loaded."""
        from models.training.trainer import load_backtest_config

        config = load_backtest_config()
        assert config is not None
        assert "splits" in config
        assert isinstance(config["splits"]["train_seasons"], list)

    def test_feature_table_registry_works(self, project_root):
        """Test that feature table registry functions work."""
        from features.feature_table_registry import (
            get_feature_table_path,
            validate_feature_table_exists,
        )

        # Test that registry can resolve paths
        path = get_feature_table_path("baseline")
        assert isinstance(path, Path)

        # Test validation (may fail if data not present, which is OK for smoke test)
        try:
            validate_feature_table_exists("baseline")
        except FileNotFoundError:
            # This is acceptable - data may not be present in CI
            pytest.skip("Feature table not found (data may not be present)")

    def test_pipeline_structure_valid(self):
        """Test that pipeline function has correct signature."""
        from orchestration.pipelines.phase1c_pipeline import run_phase1c
        import inspect

        sig = inspect.signature(run_phase1c)
        # Should take no required args (or minimal args)
        assert len([p for p in sig.parameters.values() if p.default == inspect.Parameter.empty]) == 0


class TestPhase1COutputs:
    """Test that Phase 1C produces expected outputs."""

    @pytest.fixture
    def project_root(self):
        return Path(__file__).parent.parent

    def test_output_directories_exist(self, project_root):
        """Test that output directories can be created."""
        artifacts_dir = project_root / "models" / "artifacts" / "nfl_baseline"
        reports_dir = project_root / "docs" / "reports"

        # Directories should exist (may be empty)
        assert artifacts_dir.parent.exists(), "Artifacts parent directory should exist"
        assert reports_dir.exists(), "Reports directory should exist"

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath(
            "data/nfl/processed/game_features_baseline.parquet"
        ).exists(),
        reason="Feature table not available - full pipeline requires data",
    )
    def test_pipeline_can_run_end_to_end(self):
        """
        Full end-to-end smoke test (only runs if data is available).

        This test verifies the complete pipeline runs without errors.
        Note: This requires the baseline feature table to be present.
        """
        from orchestration.pipelines.phase1c_pipeline import run_phase1c
        from pathlib import Path

        # Run pipeline
        results = run_phase1c()

        # Verify results structure
        assert results is not None
        assert isinstance(results, dict)
        assert len(results) > 0

        # Verify expected model results exist
        expected_models = ["logit", "gbm", "ensemble", "market_baseline"]
        for model_name in expected_models:
            assert model_name in results, f"Missing results for {model_name}"

        # Verify each model has validation and test results
        for model_name, model_results in results.items():
            assert "validation" in model_results, f"{model_name} missing validation results"
            assert "test" in model_results, f"{model_name} missing test results"

            val = model_results["validation"]
            test = model_results["test"]

            # Verify key metrics exist
            for metric in ["accuracy", "brier_score", "log_loss", "n_games"]:
                assert metric in val, f"{model_name} validation missing {metric}"
                assert metric in test, f"{model_name} test missing {metric}"

        # Verify report was generated
        report_path = Path(__file__).parent.parent / "docs" / "reports" / "nfl_baseline_phase1c.md"
        assert report_path.exists(), "Report file should be generated"

        # Verify report has content
        with open(report_path) as f:
            content = f.read()
            assert len(content) > 0, "Report should have content"
            assert "Phase 1C" in content or "Baseline Model" in content, "Report should mention Phase 1C"

        # Verify model artifacts exist
        artifacts_dir = Path(__file__).parent.parent / "models" / "artifacts" / "nfl_baseline"
        assert artifacts_dir.exists(), "Artifacts directory should exist"

        # Check for at least one model artifact
        artifact_files = list(artifacts_dir.glob("*.pkl")) + list(artifacts_dir.glob("*.json"))
        assert len(artifact_files) > 0, "Should have at least one model artifact file"

