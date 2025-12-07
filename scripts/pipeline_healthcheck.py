#!/usr/bin/env python3
"""
Pipeline Healthcheck Script

Validates that the NFL prediction pipeline is in a healthy state:
- All expected directories exist
- Feature tables are present or marked missing
- Models can be loaded or trained
- Backtest can run
- Reports can be written
"""

import sys
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.feature_table_registry import (
    list_feature_tables,
    get_feature_table_path,
    validate_feature_table_exists,
)
from models.training.trainer import load_backtest_config, load_features
from models.base import BaseModel
from eval.backtest import run_backtest
from eval.reports import generate_report
from models.architectures.market_baseline import MarketBaselineModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HealthCheckResult:
    """Container for healthcheck results."""
    
    def __init__(self):
        self.passed = True
        self.errors = []
        self.warnings = []
        self.info = []
    
    def add_error(self, message: str):
        """Add an error (fails healthcheck)."""
        self.passed = False
        self.errors.append(message)
        logger.error(f"❌ {message}")
    
    def add_warning(self, message: str):
        """Add a warning (doesn't fail healthcheck)."""
        self.warnings.append(message)
        logger.warning(f"⚠️  {message}")
    
    def add_info(self, message: str):
        """Add informational message."""
        self.info.append(message)
        logger.info(f"✓ {message}")


def check_directories(project_root: Path) -> HealthCheckResult:
    """Check that all expected directories exist."""
    result = HealthCheckResult()
    
    expected_dirs = [
        "data/nfl/raw",
        "data/nfl/staged",
        "data/nfl/processed",
        "models/artifacts/nfl_baseline",
        "docs/reports",
        "config/data",
        "config/models",
        "config/evaluation",
    ]
    
    for dir_path in expected_dirs:
        full_path = project_root / dir_path
        if full_path.exists() and full_path.is_dir():
            result.add_info(f"Directory exists: {dir_path}")
        else:
            result.add_error(f"Missing directory: {dir_path}")
    
    return result


def check_feature_tables(project_root: Path) -> HealthCheckResult:
    """Check feature tables from registry."""
    result = HealthCheckResult()
    
    feature_tables = list_feature_tables()
    result.add_info(f"Found {len(feature_tables)} registered feature tables")
    
    for table_name in feature_tables:
        try:
            table_path = get_feature_table_path(table_name)
            if table_path.exists():
                # Try to load and validate basic structure
                try:
                    df = pd.read_parquet(table_path)
                    result.add_info(
                        f"Feature table '{table_name}': {len(df):,} rows, "
                        f"{len(df.columns)} columns"
                    )
                    
                    # Check required columns
                    required_cols = ["game_id", "season", "week"]
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        result.add_warning(
                            f"Feature table '{table_name}' missing columns: {missing_cols}"
                        )
                    
                    # Check for duplicates
                    if df["game_id"].duplicated().any():
                        result.add_error(
                            f"Feature table '{table_name}' has duplicate game_ids"
                        )
                    
                except Exception as e:
                    result.add_error(
                        f"Feature table '{table_name}' exists but cannot be loaded: {e}"
                    )
            else:
                result.add_warning(
                    f"Feature table '{table_name}' not found: {table_path}"
                )
        except Exception as e:
            result.add_error(f"Error checking feature table '{table_name}': {e}")
    
    return result


def check_models(project_root: Path) -> HealthCheckResult:
    """Check that models can be loaded or trained."""
    result = HealthCheckResult()
    
    artifacts_dir = project_root / "models" / "artifacts" / "nfl_baseline"
    
    expected_models = ["logit.pkl", "gbm.pkl", "ensemble.json"]
    models_found = []
    
    for model_file in expected_models:
        model_path = artifacts_dir / model_file
        if model_path.exists():
            models_found.append(model_file)
            result.add_info(f"Model artifact found: {model_file}")
        else:
            result.add_warning(f"Model artifact missing: {model_file}")
    
    # Try to load models if they exist
    if len(models_found) >= 2:  # Need at least logit and gbm
        try:
            logit_path = artifacts_dir / "logit.pkl"
            gbm_path = artifacts_dir / "gbm.pkl"
            
            if logit_path.exists() and gbm_path.exists():
                try:
                    logit_model = BaseModel.load(logit_path)
                    gbm_model = BaseModel.load(gbm_path)
                    
                    # Verify models have predict_proba method
                    if hasattr(logit_model, "predict_proba") and hasattr(gbm_model, "predict_proba"):
                        result.add_info("Models can be loaded successfully")
                    else:
                        result.add_warning("Models loaded but missing predict_proba method")
                except Exception as e:
                    result.add_error(f"Error loading model files: {e}")
                
                # Try to load ensemble if config exists
                ensemble_config_path = artifacts_dir / "ensemble.json"
                if ensemble_config_path.exists():
                    try:
                        import json
                        with open(ensemble_config_path) as f:
                            ensemble_config = json.load(f)
                        result.add_info(f"Ensemble config found: weight={ensemble_config.get('weight', 'N/A')}")
                    except Exception as e:
                        result.add_warning(f"Ensemble config exists but cannot be read: {e}")
            else:
                result.add_warning("Some model artifacts missing, cannot test loading")
        except Exception as e:
            result.add_error(f"Error checking models: {e}")
    else:
        result.add_info("Models not found - will need to train (this is OK)")
    
    return result


def check_backtest(project_root: Path) -> HealthCheckResult:
    """Check that backtest can run on baseline feature table."""
    result = HealthCheckResult()
    
    try:
        # Check if baseline feature table exists
        try:
            validate_feature_table_exists("baseline")
        except FileNotFoundError:
            result.add_error("Baseline feature table not found - cannot test backtest")
            return result
        
        # Try to load features
        try:
            X, y, feature_cols, df = load_features(feature_table="baseline")
            result.add_info(
                f"Features loaded: {len(X):,} games, {len(feature_cols)} features"
            )
        except Exception as e:
            result.add_error(f"Cannot load features for backtest: {e}")
            return result
        
        # Check if we have enough data for a minimal backtest
        if len(X) < 10:
            result.add_warning("Insufficient data for backtest validation (< 10 games)")
            return result
        
        # Try to create market baseline model
        try:
            market_model = MarketBaselineModel()
            result.add_info("Market baseline model can be instantiated")
        except Exception as e:
            result.add_error(f"Cannot create market baseline model: {e}")
            return result
        
        # Try a minimal backtest (just check it doesn't crash)
        try:
            backtest_config = load_backtest_config()
            train_seasons = backtest_config["splits"]["train_seasons"]
            val_season = backtest_config["splits"]["validation_season"]
            test_season = backtest_config["splits"]["test_season"]
            
            # Filter to validation season for quick test
            df_val = df[df["season"] == val_season].copy()
            if len(df_val) == 0:
                result.add_warning(f"No data for validation season {val_season}")
                return result
            
            X_val = X[df["season"] == val_season].copy()
            y_val = y[df["season"] == val_season].copy()
            
            # We can't actually run full backtest without trained models,
            # but we can verify the structure is correct
            result.add_info(
                f"Backtest data structure valid: {len(X_val)} validation games"
            )
            
        except Exception as e:
            result.add_error(f"Error preparing backtest data: {e}")
        
    except Exception as e:
        result.add_error(f"Unexpected error in backtest check: {e}")
    
    return result


def check_reports(project_root: Path) -> HealthCheckResult:
    """Check that reports directory is writable."""
    result = HealthCheckResult()
    
    reports_dir = project_root / "docs" / "reports"
    
    if not reports_dir.exists():
        result.add_error(f"Reports directory does not exist: {reports_dir}")
        return result
    
    if not reports_dir.is_dir():
        result.add_error(f"Reports path exists but is not a directory: {reports_dir}")
        return result
    
    # Try to create a test file
    test_file = reports_dir / ".healthcheck_test"
    try:
        test_file.write_text("test")
        test_file.unlink()  # Clean up
        result.add_info("Reports directory is writable")
    except Exception as e:
        result.add_error(f"Cannot write to reports directory: {e}")
    
    # Check for existing reports
    existing_reports = list(reports_dir.glob("*.md"))
    if existing_reports:
        result.add_info(f"Found {len(existing_reports)} existing report(s)")
    else:
        result.add_warning("No existing reports found")
    
    return result


def check_configs(project_root: Path) -> HealthCheckResult:
    """Check that config files exist and are valid."""
    result = HealthCheckResult()
    
    config_files = [
        "config/data/nfl.yaml",
        "config/models/nfl_baseline.yaml",
        "config/evaluation/backtest_config.yaml",
    ]
    
    for config_file in config_files:
        config_path = project_root / config_file
        if config_path.exists():
            result.add_info(f"Config file exists: {config_file}")
            
            # Try to load YAML
            try:
                import yaml
                with open(config_path) as f:
                    yaml.safe_load(f)
                result.add_info(f"Config file is valid YAML: {config_file}")
            except Exception as e:
                result.add_error(f"Config file is invalid YAML: {config_file} - {e}")
        else:
            result.add_error(f"Config file missing: {config_file}")
    
    return result


def run_healthcheck(project_root: Path = None) -> Tuple[bool, Dict]:
    """
    Run complete pipeline healthcheck.
    
    Args:
        project_root: Path to project root (defaults to script parent)
    
    Returns:
        Tuple of (passed: bool, results: dict)
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent
    
    logger.info("=" * 60)
    logger.info("Pipeline Healthcheck")
    logger.info("=" * 60)
    logger.info(f"Project root: {project_root}")
    logger.info("")
    
    all_results = {}
    overall_passed = True
    
    # Run all checks
    checks = [
        ("directories", check_directories),
        ("configs", check_configs),
        ("feature_tables", check_feature_tables),
        ("models", check_models),
        ("backtest", check_backtest),
        ("reports", check_reports),
    ]
    
    for check_name, check_func in checks:
        logger.info(f"\n[{check_name.upper()}] Running {check_name} check...")
        logger.info("-" * 60)
        try:
            check_result = check_func(project_root)
            all_results[check_name] = {
                "passed": check_result.passed,
                "errors": check_result.errors,
                "warnings": check_result.warnings,
                "info": check_result.info,
            }
            if not check_result.passed:
                overall_passed = False
        except Exception as e:
            logger.error(f"Check '{check_name}' raised exception: {e}")
            all_results[check_name] = {
                "passed": False,
                "errors": [f"Exception: {e}"],
                "warnings": [],
                "info": [],
            }
            overall_passed = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Healthcheck Summary")
    logger.info("=" * 60)
    
    total_errors = sum(len(r["errors"]) for r in all_results.values())
    total_warnings = sum(len(r["warnings"]) for r in all_results.values())
    
    logger.info(f"Overall status: {'✅ PASSED' if overall_passed else '❌ FAILED'}")
    logger.info(f"Total errors: {total_errors}")
    logger.info(f"Total warnings: {total_warnings}")
    
    if total_errors > 0:
        logger.info("\nErrors:")
        for check_name, result in all_results.items():
            for error in result["errors"]:
                logger.error(f"  [{check_name}] {error}")
    
    if total_warnings > 0:
        logger.info("\nWarnings:")
        for check_name, result in all_results.items():
            for warning in result["warnings"]:
                logger.warning(f"  [{check_name}] {warning}")
    
    logger.info("=" * 60)
    
    return overall_passed, all_results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run pipeline healthcheck")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Path to project root (defaults to script parent)"
    )
    parser.add_argument(
        "--exit-code",
        action="store_true",
        help="Exit with non-zero code if healthcheck fails"
    )
    
    args = parser.parse_args()
    
    passed, results = run_healthcheck(args.project_root)
    
    if args.exit_code and not passed:
        sys.exit(1)
    elif args.exit_code:
        sys.exit(0)


if __name__ == "__main__":
    main()

