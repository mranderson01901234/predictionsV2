"""
Phase 1D Complete Pipeline

Runs sanity check evaluation with market baseline and season-by-season analysis.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.training.trainer import run_training_pipeline, load_backtest_config
from models.architectures.market_baseline import MarketBaselineModel
from eval.backtest import run_backtest, run_season_by_season_analysis
from eval.reports import generate_phase1d_report
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_phase1d():
    """Run complete Phase 1D pipeline."""
    logger.info("=" * 60)
    logger.info("Phase 1D: Baseline Model Sanity Check & Market Comparison")
    logger.info("=" * 60)
    
    # Load config
    backtest_config = load_backtest_config()
    train_seasons = backtest_config["splits"]["train_seasons"]
    validation_season = backtest_config["splits"]["validation_season"]
    test_season = backtest_config["splits"]["test_season"]
    edge_thresholds = backtest_config["roi"]["edge_thresholds"]
    
    # Step 1: Load trained models (or train if needed)
    logger.info("\n[Step 1/4] Loading/Training Models...")
    try:
        # Try loading existing models
        from models.base import BaseModel
        logit_path = Path(__file__).parent.parent.parent / "models" / "artifacts" / "nfl_baseline" / "logit.pkl"
        gbm_path = Path(__file__).parent.parent.parent / "models" / "artifacts" / "nfl_baseline" / "gbm.pkl"
        
        if logit_path.exists() and gbm_path.exists():
            logit_model = BaseModel.load(logit_path)
            gbm_model = BaseModel.load(gbm_path)
            
            # Load ensemble config
            import json
            ensemble_config_path = Path(__file__).parent.parent.parent / "models" / "artifacts" / "nfl_baseline" / "ensemble.json"
            with open(ensemble_config_path) as f:
                ensemble_config = json.load(f)
            
            from models.architectures.ensemble import EnsembleModel
            ensemble_model = EnsembleModel(logit_model, gbm_model, weight=ensemble_config["weight"])
            
            logger.info("Loaded existing trained models")
        else:
            raise FileNotFoundError("Models not found")
    except Exception as e:
        logger.info(f"Models not found or error loading: {e}")
        logger.info("Training new models...")
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
        ) = run_training_pipeline()
    
    # Step 2: Create market baseline model
    logger.info("\n[Step 2/4] Creating Market Baseline Model...")
    market_model = MarketBaselineModel()
    
    # Load full dataframe
    features_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "nfl"
        / "processed"
        / "game_features_baseline.parquet"
    )
    df_full = pd.read_parquet(features_path)
    
    # Split dataframe
    df_val = df_full[df_full["season"] == validation_season].copy()
    df_test = df_full[df_full["season"] == test_season].copy()
    
    # Load features for validation and test
    from models.training.trainer import load_features
    X_full, y_full, feature_cols, _ = load_features()  # Ignore df_full, already have it
    
    X_val = X_full[df_full["season"] == validation_season].copy()
    y_val = y_full[df_full["season"] == validation_season].copy()
    X_test = X_full[df_full["season"] == test_season].copy()
    y_test = y_full[df_full["season"] == test_season].copy()
    
    # Step 3: Run backtest with market baseline
    logger.info("\n[Step 3/4] Running Backtest with Market Baseline...")
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
    
    # Step 4: Season-by-season analysis
    logger.info("\n[Step 4/4] Running Season-by-Season Analysis...")
    
    # Analyze seasons from 2018 onwards (after training period)
    test_seasons_for_analysis = [2018, 2019, 2020, 2021, 2022, 2023]
    test_seasons_for_analysis = [s for s in test_seasons_for_analysis if s not in train_seasons]
    
    season_analysis = run_season_by_season_analysis(
        logit_model,
        market_model,
        X_full,
        y_full,
        df_full,
        train_seasons,
        test_seasons_for_analysis,
        edge_threshold=0.03,
    )
    
    # Step 5: Generate Report
    logger.info("\n[Step 5/5] Generating Phase 1D Report...")
    report_path = (
        Path(__file__).parent.parent.parent
        / "docs"
        / "reports"
        / "nfl_baseline_phase1d.md"
    )
    
    generate_phase1d_report(
        results,
        season_analysis,
        report_path,
        train_seasons,
        validation_season,
        test_season,
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("Phase 1D Complete!")
    logger.info("=" * 60)
    logger.info(f"\nReport saved to: {report_path}")
    
    return results, season_analysis


if __name__ == "__main__":
    run_phase1d()

