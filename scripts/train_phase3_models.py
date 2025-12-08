"""
Train Models with Phase 1-3 Features

This script:
1. Generates comprehensive features (if needed)
2. Trains ensemble model with all Phase 1-3 features
3. Applies calibration
4. Evaluates on validation and test sets
5. Saves trained models

Usage:
    python scripts/train_phase3_models.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.nfl.generate_all_features import generate_all_features
from models.training.trainer import (
    load_features,
    split_by_season,
    train_stacking_ensemble,
    load_config,
)
from models.calibration import CalibratedModel
from models.architectures.stacking_ensemble import StackingEnsemble
from eval.metrics import accuracy, brier_score, log_loss
from features.feature_table_registry import validate_feature_table_exists

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def ensure_features_exist(feature_table: str = "phase3"):
    """Ensure feature table exists, generate if needed."""
    try:
        validate_feature_table_exists(feature_table)
        logger.info(f"Feature table '{feature_table}' exists")
    except FileNotFoundError:
        logger.info(f"Feature table '{feature_table}' not found, generating...")
        generate_all_features()
        logger.info("Feature generation complete")


def train_phase3_ensemble(
    feature_table: str = "phase3",
    train_seasons: list = None,
    val_season: int = 2023,
    test_season: int = 2024,
    artifacts_dir: Path = None,
) -> dict:
    """
    Train stacking ensemble with Phase 1-3 features.
    
    Args:
        feature_table: Feature table name
        train_seasons: List of training seasons (default: 2015-2022)
        val_season: Validation season
        test_season: Test season
        artifacts_dir: Directory to save models
    
    Returns:
        Dictionary with training results
    """
    logger.info("=" * 60)
    logger.info("PHASE 3 MODEL TRAINING")
    logger.info("=" * 60)
    
    # Ensure features exist
    ensure_features_exist(feature_table)
    
    # Load features
    logger.info(f"Loading features from table: {feature_table}")
    X, y, feature_cols, games_df = load_features(feature_table=feature_table)
    
    logger.info(f"Loaded {len(X)} games with {len(feature_cols)} features")
    logger.info(f"Feature columns: {feature_cols[:10]}... ({len(feature_cols)} total)")
    
    # Split by season
    if train_seasons is None:
        train_seasons = list(range(2015, 2023))
    
    train_mask = games_df['season'].isin(train_seasons)
    val_mask = games_df['season'] == val_season
    test_mask = games_df['season'] == test_season
    
    X_train = X[train_mask].copy()
    y_train = y[train_mask].copy()
    games_train = games_df[train_mask].copy()
    
    X_val = X[val_mask].copy()
    y_val = y[val_mask].copy()
    games_val = games_df[val_mask].copy()
    
    X_test = X[test_mask].copy()
    y_test = y[test_mask].copy()
    games_test = games_df[test_mask].copy()
    
    logger.info(f"Train: {len(X_train)} games ({min(train_seasons)}-{max(train_seasons)})")
    logger.info(f"Val: {len(X_val)} games ({val_season})")
    logger.info(f"Test: {len(X_test)} games ({test_season})")
    
    # Set up artifacts directory
    if artifacts_dir is None:
        artifacts_dir = Path(__file__).parent.parent / "models" / "artifacts" / "nfl_phase3"
    
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Train ensemble
    logger.info("=" * 60)
    logger.info("Training Stacking Ensemble")
    logger.info("=" * 60)
    
    # Load default config or create minimal config
    try:
        config = load_config()
    except:
        # Create minimal config for ensemble training
        config = {
            'base_models': {
                'logistic': {'type': 'logistic_regression'},
                'gbm': {'type': 'gradient_boosting'},
            },
            'meta_model': {'type': 'logistic'},
            'stacking': {'include_features': False},
            'random_state': 42,
        }
    
    ensemble = train_stacking_ensemble(
        X_train=X_train[feature_cols],
        y_train=y_train,
        X_val=X_val[feature_cols],
        y_val=y_val,
        config=config,
        artifacts_dir=artifacts_dir,
    )
    
    # Save ensemble
    ensemble_path = artifacts_dir / "ensemble.pkl"
    ensemble.save(ensemble_path)
    logger.info(f"Saved ensemble to {ensemble_path}")
    
    # Apply calibration
    logger.info("=" * 60)
    logger.info("Applying Calibration")
    logger.info("=" * 60)
    
    # Get validation predictions for calibration
    val_probs_raw = ensemble.predict_proba(X_val[feature_cols])
    
    # Fit calibrator
    calibrated_model = CalibratedModel(
        base_model=ensemble,
        method="isotonic",  # Best for most cases
    )
    calibrated_model.fit(
        X_train[feature_cols],
        y_train,
        X_cal=X_val[feature_cols],
        y_cal=y_val,
    )
    
    # Save calibrated model
    calibrated_path = artifacts_dir / "ensemble_calibrated.pkl"
    calibrated_model.save(calibrated_path)
    logger.info(f"Saved calibrated ensemble to {calibrated_path}")
    
    # Evaluate
    logger.info("=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)
    
    # Validation set
    val_probs_cal = calibrated_model.predict_proba(X_val[feature_cols])
    val_preds = (val_probs_cal >= 0.5).astype(int)
    
    val_acc = accuracy(y_val, val_preds)
    val_brier = brier_score(y_val, val_probs_cal)
    val_logloss = log_loss(y_val, val_probs_cal)
    
    logger.info(f"Validation Set ({val_season}):")
    logger.info(f"  Accuracy: {val_acc:.4f}")
    logger.info(f"  Brier Score: {val_brier:.4f}")
    logger.info(f"  Log Loss: {val_logloss:.4f}")
    
    # Test set
    test_probs_cal = calibrated_model.predict_proba(X_test[feature_cols])
    test_preds = (test_probs_cal >= 0.5).astype(int)
    
    test_acc = accuracy(y_test, test_preds)
    test_brier = brier_score(y_test, test_probs_cal)
    test_logloss = log_loss(y_test, test_probs_cal)
    
    logger.info(f"Test Set ({test_season}):")
    logger.info(f"  Accuracy: {test_acc:.4f}")
    logger.info(f"  Brier Score: {test_brier:.4f}")
    logger.info(f"  Log Loss: {test_logloss:.4f}")
    
    # Compile results
    results = {
        'feature_table': feature_table,
        'train_seasons': train_seasons,
        'val_season': val_season,
        'test_season': test_season,
        'n_features': len(feature_cols),
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'validation': {
            'accuracy': float(val_acc),
            'brier_score': float(val_brier),
            'log_loss': float(val_logloss),
        },
        'test': {
            'accuracy': float(test_acc),
            'brier_score': float(test_brier),
            'log_loss': float(test_logloss),
        },
        'model_paths': {
            'ensemble': str(ensemble_path),
            'calibrated': str(calibrated_path),
        },
    }
    
    # Save results
    results_path = artifacts_dir / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Saved results to {results_path}")
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train models with Phase 1-3 features")
    parser.add_argument('--feature-table', type=str, default='phase3', help='Feature table name')
    parser.add_argument('--generate-features', action='store_true', help='Generate features before training')
    parser.add_argument('--artifacts-dir', type=str, help='Directory to save models')
    
    args = parser.parse_args()
    
    if args.generate_features:
        logger.info("Generating features...")
        generate_all_features()
    
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else None
    
    results = train_phase3_ensemble(
        feature_table=args.feature_table,
        artifacts_dir=artifacts_dir,
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Validation Accuracy: {results['validation']['accuracy']:.2%}")
    logger.info(f"Test Accuracy: {results['test']['accuracy']:.2%}")
    logger.info(f"Models saved to: {results['model_paths']['ensemble']}")


if __name__ == "__main__":
    main()

