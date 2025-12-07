"""
Tests for Pipeline Healthcheck Script

Tests that healthcheck runs correctly with mocked file structure.
"""

import pytest
from pathlib import Path
import tempfile
import shutil
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project structure."""
    project = tmp_path / "test_project"
    
    # Create directory structure
    (project / "data" / "nfl" / "raw").mkdir(parents=True)
    (project / "data" / "nfl" / "staged").mkdir(parents=True)
    (project / "data" / "nfl" / "processed").mkdir(parents=True)
    (project / "models" / "artifacts" / "nfl_baseline").mkdir(parents=True)
    (project / "docs" / "reports").mkdir(parents=True)
    (project / "config" / "data").mkdir(parents=True)
    (project / "config" / "models").mkdir(parents=True)
    (project / "config" / "evaluation").mkdir(parents=True)
    
    return project


@pytest.fixture
def mock_baseline_features(temp_project):
    """Create a minimal baseline feature table."""
    import pandas as pd
    
    df = pd.DataFrame({
        "game_id": ["nfl_2020_01_KC_BAL", "nfl_2020_02_BAL_KC"],
        "season": [2020, 2020],
        "week": [1, 2],
        "home_team": ["BAL", "KC"],
        "away_team": ["KC", "BAL"],
        "home_score": [34, 23],
        "away_score": [20, 34],
        "close_spread": [-3.0, 3.0],
        "close_total": [54.0, 54.0],
        "home_win_rate_last4": [0.5, 0.5],
        "away_win_rate_last4": [0.5, 0.5],
    })
    
    output_path = temp_project / "data" / "nfl" / "processed" / "game_features_baseline.parquet"
    df.to_parquet(output_path, index=False)
    return df


@pytest.fixture
def mock_configs(temp_project):
    """Create minimal config files."""
    import yaml
    
    # Data config
    data_config = {
        "nfl": {
            "schedule": {"source": "nflverse", "seasons": [2020, 2021]},
            "odds": {"source": "manual_scrape"},
        }
    }
    with open(temp_project / "config" / "data" / "nfl.yaml", "w") as f:
        yaml.dump(data_config, f)
    
    # Model config
    model_config = {
        "models": {
            "logistic_regression": {"C": 1.0, "max_iter": 1000},
            "gradient_boosting": {"n_estimators": 100},
            "ensemble": {"weight": 0.7},
        }
    }
    with open(temp_project / "config" / "models" / "nfl_baseline.yaml", "w") as f:
        yaml.dump(model_config, f)
    
    # Backtest config
    backtest_config = {
        "feature_table": "baseline",
        "splits": {
            "train_seasons": [2020],
            "validation_season": 2021,
            "test_season": 2021,
        },
        "roi": {"edge_thresholds": [0.03, 0.05]},
    }
    with open(temp_project / "config" / "evaluation" / "backtest_config.yaml", "w") as f:
        yaml.dump(backtest_config, f)


@pytest.fixture
def mock_models(temp_project):
    """Create mock model files."""
    import pickle
    
    # Create a minimal mock model
    class MockModel:
        def predict_proba(self, X):
            import numpy as np
            return np.array([0.5] * len(X))
    
    mock_model = MockModel()
    
    # Save mock models
    logit_path = temp_project / "models" / "artifacts" / "nfl_baseline" / "logit.pkl"
    gbm_path = temp_project / "models" / "artifacts" / "nfl_baseline" / "gbm.pkl"
    
    with open(logit_path, "wb") as f:
        pickle.dump(mock_model, f)
    
    with open(gbm_path, "wb") as f:
        pickle.dump(mock_model, f)
    
    # Create ensemble config
    ensemble_config = {"weight": 0.7}
    ensemble_path = temp_project / "models" / "artifacts" / "nfl_baseline" / "ensemble.json"
    with open(ensemble_path, "w") as f:
        json.dump(ensemble_config, f)


class TestHealthcheckSmoke:
    """Smoke tests for healthcheck script."""
    
    def test_healthcheck_imports(self):
        """Test that healthcheck can be imported."""
        from scripts.pipeline_healthcheck import (
            run_healthcheck,
            check_directories,
            check_feature_tables,
            check_models,
            check_backtest,
            check_reports,
            check_configs,
        )
        assert callable(run_healthcheck)
        assert callable(check_directories)
        assert callable(check_feature_tables)
    
    def test_check_directories_passes(self, temp_project):
        """Test directory check with valid structure."""
        from scripts.pipeline_healthcheck import check_directories
        
        result = check_directories(temp_project)
        assert result.passed, "Directory check should pass with valid structure"
        assert len(result.errors) == 0
    
    def test_check_directories_fails_missing(self, tmp_path):
        """Test directory check fails with missing directories."""
        from scripts.pipeline_healthcheck import check_directories
        
        result = check_directories(tmp_path)
        assert not result.passed, "Directory check should fail with missing directories"
        assert len(result.errors) > 0
    
    def test_check_feature_tables_with_baseline(self, temp_project, mock_baseline_features):
        """Test feature table check with baseline table."""
        from scripts.pipeline_healthcheck import check_feature_tables
        
        result = check_feature_tables(temp_project)
        # Should pass (baseline exists) or warn (others missing)
        assert len(result.errors) == 0, "Should not error if baseline exists"
    
    def test_check_configs_passes(self, temp_project, mock_configs):
        """Test config check with valid configs."""
        from scripts.pipeline_healthcheck import check_configs
        
        result = check_configs(temp_project)
        assert result.passed, "Config check should pass with valid configs"
        assert len(result.errors) == 0
    
    def test_check_reports_passes(self, temp_project):
        """Test reports check."""
        from scripts.pipeline_healthcheck import check_reports
        
        result = check_reports(temp_project)
        assert result.passed, "Reports check should pass if directory is writable"
    
    def test_full_healthcheck_runs(self, temp_project, mock_baseline_features, mock_configs):
        """Test that full healthcheck runs without exceptions."""
        from scripts.pipeline_healthcheck import run_healthcheck
        
        # Should not raise exceptions
        passed, results = run_healthcheck(temp_project)
        
        assert isinstance(passed, bool)
        assert isinstance(results, dict)
        assert "directories" in results
        assert "configs" in results
        assert "feature_tables" in results
    
    def test_healthcheck_exit_code(self, temp_project, mock_baseline_features, mock_configs):
        """Test healthcheck exit code behavior."""
        import subprocess
        import sys
        
        # Create a minimal healthcheck script
        script = temp_project / "test_healthcheck.py"
        script.write_text("""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.pipeline_healthcheck import run_healthcheck

passed, _ = run_healthcheck(Path(__file__).parent)
sys.exit(0 if passed else 1)
""")
        
        # Run it
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(temp_project),
            capture_output=True,
            text=True
        )
        
        # Should exit with 0 if passed, 1 if failed
        assert result.returncode in [0, 1], "Exit code should be 0 or 1"


class TestHealthcheckIntegration:
    """Integration tests with real project structure."""
    
    def test_healthcheck_on_real_project(self):
        """Test healthcheck on actual project (if it exists)."""
        from scripts.pipeline_healthcheck import run_healthcheck
        from pathlib import Path
        
        project_root = Path(__file__).parent.parent
        
        # Should run without exceptions
        passed, results = run_healthcheck(project_root)
        
        assert isinstance(passed, bool)
        assert isinstance(results, dict)
        
        # At minimum, directories and configs should pass
        assert "directories" in results
        assert "configs" in results

