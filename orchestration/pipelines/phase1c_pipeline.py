"""
Phase 1C Complete Pipeline

Runs training, evaluation, and report generation for baseline models.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.training.trainer import run_training_pipeline, load_backtest_config
from models.architectures.market_baseline import MarketBaselineModel
from eval.backtest import run_backtest
from eval.reports import generate_report
from features.feature_table_registry import get_feature_table_path, validate_feature_table_exists
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_phase1c():
    """Run complete Phase 1C pipeline."""
    logger.info("=" * 60)
    logger.info("Phase 1C: Baseline Model Training & Evaluation")
    logger.info("=" * 60)
    
    # Load config
    backtest_config = load_backtest_config()
    feature_table = backtest_config.get("feature_table", "baseline")
    train_seasons = backtest_config["splits"]["train_seasons"]
    validation_season = backtest_config["splits"]["validation_season"]
    test_season = backtest_config["splits"]["test_season"]
    edge_thresholds = backtest_config["roi"]["edge_thresholds"]
    
    # Log configuration
    logger.info(f"\nConfiguration:")
    logger.info(f"  Feature table: {feature_table}")
    logger.info(f"  Train seasons: {train_seasons}")
    logger.info(f"  Validation season: {validation_season}")
    logger.info(f"  Test season: {test_season}")
    logger.info(f"  Edge thresholds: {edge_thresholds}")
    
    # Step 1: Training
    logger.info("\n[Step 1/3] Training Models...")
    logger.info("Training logistic regression, gradient boosting, and ensemble models...")
    (
        logit_model,
        gbm_model,
        ensemble_model,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        df_full,
    ) = run_training_pipeline()
    
    logger.info("✓ Models trained successfully")
    logger.info(f"  Training set: {len(X_train)} games")
    logger.info(f"  Validation set: {len(X_val)} games")
    logger.info(f"  Test set: {len(X_test)} games")
    logger.info(f"  Full dataframe: {len(df_full)} games (reused from training, no reload)")
    
    # Create market baseline model
    logger.info("\nCreating market baseline model for comparison...")
    market_model = MarketBaselineModel()
    logger.info("✓ Market baseline model created")
    
    # Split dataframe (using already-loaded df_full, no reload)
    df_val = df_full[df_full["season"] == validation_season].copy()
    df_test = df_full[df_full["season"] == test_season].copy()
    
    logger.info(f"  Validation set: {len(df_val)} games")
    logger.info(f"  Test set: {len(df_test)} games")
    
    # Step 2: Evaluation
    logger.info("\n[Step 2/3] Running Backtest Evaluation...")
    results = run_backtest(
        logit_model,
        gbm_model,
        ensemble_model,
        market_model,
        X_val,
        y_val,
        df_val,
        X_test,
        y_test,
        df_test,
        edge_thresholds,
    )
    
    # Log backtest results summary
    logger.info("\n" + "=" * 60)
    logger.info("Backtest Results Summary")
    logger.info("=" * 60)
    
    for model_name, model_results in results.items():
        val = model_results["validation"]
        test = model_results["test"]
        logger.info(f"\n{model_name.upper().replace('_', ' ')}:")
        logger.info(f"  Validation: Accuracy={val['accuracy']:.4f}, Brier={val['brier_score']:.4f}, Log Loss={val['log_loss']:.4f}")
        logger.info(f"  Test:       Accuracy={test['accuracy']:.4f}, Brier={test['brier_score']:.4f}, Log Loss={test['log_loss']:.4f}")
        
        # Log ROI results
        for roi_key, roi_data in test["roi_results"].items():
            threshold = roi_data["edge_threshold"]
            logger.info(f"    ROI (edge >= {threshold:.0%}): {roi_data['roi']:.2%} ({roi_data['n_bets']} bets, {roi_data['win_rate']:.2%} win rate)")
    
    # Step 3: Generate Report
    logger.info("\n[Step 3/3] Generating Report...")
    report_path = (
        Path(__file__).parent.parent.parent
        / "docs"
        / "reports"
        / "nfl_baseline_phase1c.md"
    )
    
    generate_report(
        results,
        report_path,
        train_seasons,
        validation_season,
        test_season,
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("Phase 1C Complete!")
    logger.info("=" * 60)
    logger.info(f"\nReport saved to: {report_path}")
    logger.info(f"\nModels saved to: models/artifacts/nfl_baseline/")
    
    return results


if __name__ == "__main__":
    run_phase1c()

