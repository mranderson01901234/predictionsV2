"""
Train Models on All Completed Data Through 2025

Updates training to include all completed games through current date in 2025.
Then makes a real-world prediction for tonight's game.

Usage:
    python scripts/train_with_all_data.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Optional
import pytz

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.nfl.generate_all_features import generate_all_features
from models.training.trainer import (
    load_features,
    split_by_season,
    train_stacking_ensemble,
)
from models.calibration import CalibratedModel
from models.training.trainer import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_completed_games_cutoff() -> datetime:
    """Get cutoff date for completed games (today, before tonight's game)."""
    et = pytz.timezone('America/New_York')
    now = datetime.now(et)
    # Use games completed before today (or before 8:20 PM today)
    cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return cutoff


# Removed filter_completed_games function - filtering done inline


def train_on_all_data(
    feature_table: str = "phase3",
    artifacts_dir: Optional[Path] = None,
) -> dict:
    """
    Train models on all completed data through 2025.
    
    Args:
        feature_table: Feature table name to use
        artifacts_dir: Directory to save artifacts
    
    Returns:
        Dictionary with training results
    """
    logger.info("=" * 60)
    logger.info("TRAINING ON ALL COMPLETED DATA")
    logger.info("=" * 60)
    
    # Get cutoff date
    cutoff = get_completed_games_cutoff()
    logger.info(f"Cutoff date: {cutoff.date()} (only games completed before this date)")
    
    # Load features
    logger.info(f"Loading features from table: {feature_table}")
    X, y, feature_cols, games_df_features = load_features(feature_table=feature_table)
    
    # games_df_features has all metadata including game_id, scores, season, etc.
    # Merge X features with metadata
    features_df = games_df_features.copy()
    for col in feature_cols:
        features_df[col] = X[col].values
    features_df['home_win'] = y.values
    
    # Merge with games data to get dates
    games_path = Path(__file__).parent.parent / "data" / "nfl" / "staged" / "games.parquet"
    games_df = pd.read_parquet(games_path)
    
    # Merge to get game dates
    features_with_dates = features_df.merge(
        games_df[['game_id', 'date']].rename(columns={'date': 'gameday'}),
        on='game_id',
        how='left'
    )
    
    # Filter to completed games (must have scores)
    completed_features = features_with_dates[
        features_with_dates['home_score'].notna() &
        features_with_dates['away_score'].notna()
    ].copy()
    
    # Further filter by date if gameday exists
    if 'gameday' in completed_features.columns:
        if completed_features['gameday'].dtype == 'object':
            completed_features['gameday'] = pd.to_datetime(completed_features['gameday'])
        completed_features = completed_features[
            completed_features['gameday'] < pd.Timestamp(cutoff.date())
        ].copy()
    
    logger.info(f"Filtered to {len(completed_features)} completed games")
    
    logger.info(f"Training on {len(completed_features)} completed games")
    
    # Get feature columns (exclude metadata columns)
    feature_cols = [c for c in completed_features.columns if c.startswith(('home_', 'away_')) and c not in ['home_team', 'away_team', 'home_score', 'away_score', 'gameday', 'game_id', 'season', 'home_win']]
    
    # Create target
    y = (completed_features['home_score'] > completed_features['away_score']).astype(int)
    X = completed_features[feature_cols].fillna(0)
    
    # Split: Use last season for validation/test
    # Train: 2015-2023, Val: 2024, Test: 2025 (completed games only)
    train_seasons = list(range(2015, 2024))
    val_season = 2024
    test_season = 2025
    
    X_train, y_train, X_val, y_val, X_test, y_test = split_by_season(
        X, y, completed_features,
        train_seasons=train_seasons,
        validation_season=val_season,
        test_season=test_season,
    )
    
    logger.info(f"Train: {len(X_train)} games ({min(train_seasons)}-{max(train_seasons)})")
    logger.info(f"Val: {len(X_val)} games ({val_season})")
    logger.info(f"Test: {len(X_test)} games ({test_season} - completed only)")
    
    # Set up artifacts directory
    if artifacts_dir is None:
        artifacts_dir = Path(__file__).parent.parent / "models" / "artifacts" / "nfl_all_data"
    
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Train ensemble
    logger.info("=" * 60)
    logger.info("Training Stacking Ensemble")
    logger.info("=" * 60)
    
    # Load default config or create minimal config
    try:
        config = load_config()
    except:
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
    
    val_probs_raw = ensemble.predict_proba(X_val[feature_cols])
    calibrated = CalibratedModel(ensemble, method='isotonic')
    calibrated.fit(X_val[feature_cols], y_val)
    
    calibrated_path = artifacts_dir / "ensemble_calibrated.pkl"
    calibrated.save(calibrated_path)
    logger.info(f"Saved calibrated ensemble to {calibrated_path}")
    
    # Evaluate
    logger.info("=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)
    
    val_probs = calibrated.predict_proba(X_val[feature_cols])
    test_probs = calibrated.predict_proba(X_test[feature_cols])
    
    val_accuracy = (val_probs >= 0.5) == y_val
    test_accuracy = (test_probs >= 0.5) == y_test
    
    val_brier = np.mean((val_probs - y_val) ** 2)
    test_brier = np.mean((test_probs - y_test) ** 2)
    
    val_logloss = -np.mean(y_val * np.log(val_probs + 1e-10) + (1 - y_val) * np.log(1 - val_probs + 1e-10))
    test_logloss = -np.mean(y_test * np.log(test_probs + 1e-10) + (1 - y_test) * np.log(1 - test_probs + 1e-10))
    
    logger.info(f"Validation Set ({val_season}):")
    logger.info(f"  Accuracy: {val_accuracy.mean():.4f}")
    logger.info(f"  Brier Score: {val_brier:.4f}")
    logger.info(f"  Log Loss: {val_logloss:.4f}")
    
    logger.info(f"Test Set ({test_season}):")
    logger.info(f"  Accuracy: {test_accuracy.mean():.4f}")
    logger.info(f"  Brier Score: {test_brier:.4f}")
    logger.info(f"  Log Loss: {test_logloss:.4f}")
    
    # Save results
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
            'accuracy': float(val_accuracy.mean()),
            'brier_score': float(val_brier),
            'log_loss': float(val_logloss),
        },
        'test': {
            'accuracy': float(test_accuracy.mean()),
            'brier_score': float(test_brier),
            'log_loss': float(test_logloss),
        },
        'model_paths': {
            'ensemble': str(ensemble_path),
            'calibrated': str(calibrated_path),
        },
    }
    
    results_path = artifacts_dir / "training_results.json"
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {results_path}")
    
    return results


if __name__ == "__main__":
    try:
        results = train_on_all_data()
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Model saved to: {results['model_paths']['calibrated']}")
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        sys.exit(1)

