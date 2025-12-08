"""
Train Final Production Model

Comprehensive training pipeline that:
1. Validates data completeness
2. Optionally runs walk-forward validation
3. Trains on maximum historical data
4. Applies proper calibration
5. Saves production-ready artifacts

Usage:
    # Full pipeline with walk-forward validation
    python scripts/train_final_model.py \
        --features data/nfl/processed/game_features_phase3.parquet \
        --train-start 2015 \
        --train-end 2023 \
        --val-season 2024 \
        --output models/artifacts/

    # Quick training (skip walk-forward)
    python scripts/train_final_model.py \
        --features data/nfl/processed/game_features_phase3.parquet \
        --train-start 2019 \
        --train-end 2023 \
        --val-season 2024 \
        --no-walk-forward

    # With specific model type
    python scripts/train_final_model.py \
        --features data/nfl/processed/game_features_phase3.parquet \
        --model ensemble \
        --calibration isotonic
"""

import argparse
import logging
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import sys

import pandas as pd
import numpy as np
import joblib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_data_completeness(
    features_df: pd.DataFrame,
    required_seasons: List[int],
) -> Dict[str, Any]:
    """
    Validate that we have complete data for all required seasons.

    Returns:
        Dictionary with validation results
    """
    logger.info("Validating data completeness...")

    results = {
        'seasons': {},
        'is_complete': True,
        'warnings': [],
    }

    for season in required_seasons:
        season_df = features_df[features_df['season'] == season]
        n_games = len(season_df)

        # Expected: 267-272 games per season
        expected_min = 250
        expected_max = 300

        if n_games == 0:
            results['is_complete'] = False
            results['warnings'].append(f"Season {season}: No games found!")
        elif n_games < expected_min:
            results['warnings'].append(f"Season {season}: Only {n_games} games (expected ~270)")
        elif n_games > expected_max:
            results['warnings'].append(f"Season {season}: {n_games} games (more than expected)")

        # Check for missing values
        missing_pct = season_df.isnull().mean().mean() * 100

        results['seasons'][season] = {
            'n_games': n_games,
            'missing_pct': missing_pct,
        }

        logger.info(f"Season {season}: {n_games} games, {missing_pct:.1f}% missing")

    # Check QB deviation features if present
    qb_cols = [c for c in features_df.columns if 'qb_' in c and
               ('zscore' in c or 'vs_career' in c or 'career_' in c)]
    if qb_cols:
        qb_missing = features_df[qb_cols].isnull().mean().mean() * 100
        results['qb_features'] = {
            'n_features': len(qb_cols),
            'missing_pct': qb_missing,
        }
        logger.info(f"QB deviation features: {len(qb_cols)} features, {qb_missing:.1f}% missing")

    for warning in results['warnings']:
        logger.warning(warning)

    return results


def get_feature_columns(
    df: pd.DataFrame,
    include_qb_deviation: bool = True,
) -> List[str]:
    """
    Get list of feature columns, excluding metadata and targets.

    Args:
        df: DataFrame with all columns
        include_qb_deviation: Whether to include QB deviation features

    Returns:
        List of feature column names
    """
    exclude_cols = [
        'game_id', 'season', 'week', 'date', 'game_type',
        'home_team', 'away_team', 'home_score', 'away_score',
        'home_win', 'close_spread', 'close_total',
        'open_spread', 'open_total',
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    if not include_qb_deviation:
        qb_cols = [c for c in feature_cols if 'qb_' in c and
                   ('zscore' in c or 'vs_career' in c or 'career_' in c)]
        feature_cols = [c for c in feature_cols if c not in qb_cols]

    return feature_cols


def create_model(
    model_type: str,
    config: Optional[Dict] = None,
):
    """
    Create a model instance based on type.

    Args:
        model_type: Type of model ('gbm', 'lr', 'ensemble', 'ft_transformer', 'tabnet')
        config: Optional model configuration

    Returns:
        Model instance
    """
    config = config or {}

    if model_type in ['gbm', 'gradient_boosting', 'xgboost']:
        from models.architectures.gradient_boosting import GradientBoostingModel
        return GradientBoostingModel(**config)

    elif model_type in ['lr', 'logistic', 'logistic_regression']:
        from models.architectures.logistic_regression import LogisticRegressionModel
        return LogisticRegressionModel(**config)

    elif model_type in ['ensemble', 'stacking_ensemble', 'stacking']:
        from models.architectures.stacking_ensemble import StackingEnsemble
        from models.architectures.logistic_regression import LogisticRegressionModel
        from models.architectures.gradient_boosting import GradientBoostingModel

        # Default base models
        if 'base_models' not in config:
            config['base_models'] = {
                'logistic': LogisticRegressionModel(),
                'gbm': GradientBoostingModel(),
            }
        return StackingEnsemble(**config)

    elif model_type == 'ft_transformer':
        from models.architectures.ft_transformer import FTTransformerModel
        default_config = {'epochs': 100, 'patience': 15, 'd_model': 64, 'n_heads': 4}
        default_config.update(config)
        return FTTransformerModel(**default_config)

    elif model_type == 'tabnet':
        from models.architectures.tabnet import TabNetModel
        default_config = {'epochs': 100, 'patience': 15}
        default_config.update(config)
        return TabNetModel(**default_config)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_final_model(
    features_path: str,
    output_dir: str,
    train_seasons: List[int],
    val_season: int,
    model_type: str = 'gbm',
    calibration_method: str = 'isotonic',
    include_qb_deviation: bool = True,
    run_walk_forward: bool = True,
    walk_forward_start_season: Optional[int] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train final production model.

    Args:
        features_path: Path to features parquet file
        output_dir: Output directory for artifacts
        train_seasons: List of seasons to use for training
        val_season: Season to use for validation/calibration
        model_type: Model type
        calibration_method: Calibration method
        include_qb_deviation: Whether to include QB deviation features
        run_walk_forward: Whether to run walk-forward validation
        walk_forward_start_season: First test season for walk-forward

    Returns:
        Tuple of (trained_model, training_config)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info("FINAL MODEL TRAINING PIPELINE")
    logger.info(f"{'='*60}")
    logger.info(f"Output directory: {run_dir}")

    # Load features
    logger.info(f"\nLoading features from {features_path}")
    features_df = pd.read_parquet(features_path)

    # Ensure home_win target exists
    if 'home_win' not in features_df.columns:
        features_df['home_win'] = (features_df['home_score'] > features_df['away_score']).astype(int)

    logger.info(f"Total games: {len(features_df)}")
    logger.info(f"Seasons: {sorted(features_df['season'].unique())}")

    # Validate data
    all_seasons = list(set(train_seasons + [val_season]))
    validation_result = validate_data_completeness(features_df, all_seasons)

    # Get feature columns
    feature_cols = get_feature_columns(features_df, include_qb_deviation)
    logger.info(f"Using {len(feature_cols)} features")

    # Check for QB deviation features
    qb_cols = [c for c in feature_cols if 'qb_' in c and
               ('zscore' in c or 'vs_career' in c or 'career_' in c)]
    if include_qb_deviation:
        logger.info(f"QB deviation features: {len(qb_cols)}")
        if len(qb_cols) == 0:
            logger.warning("No QB deviation features found in dataset!")

    # Step 1: Walk-forward validation (optional)
    wf_summary = None
    if run_walk_forward:
        logger.info(f"\n{'='*60}")
        logger.info("STEP 1: WALK-FORWARD VALIDATION")
        logger.info(f"{'='*60}")

        from models.training.walk_forward import WalkForwardValidator, create_model_factory

        model_factory = create_model_factory(model_type)

        validator = WalkForwardValidator(
            features_df=features_df,
            model_factory=model_factory,
            min_train_seasons=4,
            calibration_method=calibration_method,
            feature_cols=feature_cols,
        )

        wf_results = validator.run_all_splits(start_test_season=walk_forward_start_season)
        wf_summary = validator.summarize_results()

        # Save walk-forward results
        wf_summary.to_csv(os.path.join(run_dir, "walk_forward_summary.csv"), index=False)

        all_wf_predictions = pd.concat([r.predictions for r in wf_results])
        all_wf_predictions.to_parquet(os.path.join(run_dir, "walk_forward_predictions.parquet"))

        logger.info(f"Walk-forward results saved to {run_dir}")

    # Step 2: Train final model on all training data
    logger.info(f"\n{'='*60}")
    logger.info("STEP 2: TRAIN FINAL MODEL")
    logger.info(f"{'='*60}")

    # Prepare data
    train_mask = features_df['season'].isin(train_seasons)
    val_mask = features_df['season'] == val_season

    train_df = features_df[train_mask].copy()
    val_df = features_df[val_mask].copy()

    logger.info(f"Training seasons: {train_seasons}")
    logger.info(f"Validation season: {val_season}")
    logger.info(f"Training games: {len(train_df)}")
    logger.info(f"Validation games: {len(val_df)}")

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['home_win']
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df['home_win']

    # Create and train model
    model = create_model(model_type)

    # Train (handle models that need validation data)
    if hasattr(model, 'fit'):
        try:
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        except TypeError:
            model.fit(X_train, y_train)

    # Step 3: Calibration
    logger.info(f"\n{'='*60}")
    logger.info("STEP 3: CALIBRATION")
    logger.info(f"{'='*60}")

    # Get uncalibrated predictions
    val_proba_uncal = model.predict_proba(X_val)
    if len(val_proba_uncal.shape) > 1:
        val_proba_uncal = val_proba_uncal[:, 1] if val_proba_uncal.shape[1] > 1 else val_proba_uncal.ravel()

    # Diagnose and compare calibration methods
    from models.calibration_diagnosis import diagnose_calibration, compare_calibration_methods

    logger.info("\nUncalibrated model calibration:")
    diagnose_calibration(y_val.values, val_proba_uncal, "Uncalibrated", print_report=True)

    # Compare calibration methods
    logger.info("\nComparing calibration methods...")
    cal_results, best_method = compare_calibration_methods(
        y_val.values, val_proba_uncal,
        methods=['platt', 'isotonic', 'temperature'],
        print_results=True
    )

    # Apply calibration
    from models.calibration import CalibratedModel

    calibrated_model = CalibratedModel(base_model=model, method=calibration_method)
    calibrated_model.fit_calibration(X_val, y_val)

    # Verify calibration
    val_proba_cal = calibrated_model.predict_proba(X_val)
    logger.info(f"\nCalibrated model ({calibration_method}):")
    diagnose_calibration(y_val.values, val_proba_cal, f"Calibrated ({calibration_method})", print_report=True)

    # Step 4: Final evaluation
    logger.info(f"\n{'='*60}")
    logger.info("STEP 4: FINAL EVALUATION")
    logger.info(f"{'='*60}")

    y_pred = (val_proba_cal >= 0.5).astype(int)

    accuracy = accuracy_score(y_val, y_pred)
    brier = brier_score_loss(y_val, val_proba_cal)
    logloss = log_loss(y_val, np.clip(val_proba_cal, 1e-7, 1-1e-7))

    logger.info(f"\nFinal Model Performance (Validation {val_season}):")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Brier Score: {brier:.4f}")
    logger.info(f"  Log Loss: {logloss:.4f}")

    # Analyze by confidence tier
    logger.info("\nAccuracy by confidence tier:")
    for low, high in [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 1.00)]:
        mask = (val_proba_cal >= low) & (val_proba_cal < high)
        if mask.sum() > 0:
            tier_acc = (y_pred[mask] == y_val.values[mask]).mean()
            logger.info(f"  {low:.0%}-{high:.0%}: {tier_acc:.3f} (n={mask.sum()})")

    # Step 5: Save artifacts
    logger.info(f"\n{'='*60}")
    logger.info("STEP 5: SAVE ARTIFACTS")
    logger.info(f"{'='*60}")

    # Save model
    model_path = os.path.join(run_dir, "final_model.pkl")
    calibrated_model.save(model_path)
    logger.info(f"Model saved: {model_path}")

    # Save feature columns
    feature_cols_path = os.path.join(run_dir, "feature_columns.json")
    with open(feature_cols_path, 'w') as f:
        json.dump(feature_cols, f, indent=2)
    logger.info(f"Feature columns saved: {feature_cols_path}")

    # Save training config
    config = {
        'features_path': features_path,
        'model_type': model_type,
        'train_seasons': train_seasons,
        'val_season': val_season,
        'n_train_games': len(train_df),
        'n_val_games': len(val_df),
        'n_features': len(feature_cols),
        'n_qb_features': len(qb_cols),
        'calibration_method': calibration_method,
        'calibration_best_method': best_method,
        'accuracy': float(accuracy),
        'brier_score': float(brier),
        'log_loss': float(logloss),
        'walk_forward_run': run_walk_forward,
        'timestamp': timestamp,
    }

    if wf_summary is not None:
        config['walk_forward_mean_accuracy'] = float(wf_summary['accuracy'].mean())
        config['walk_forward_std_accuracy'] = float(wf_summary['accuracy'].std())

    config_path = os.path.join(run_dir, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved: {config_path}")

    # Save validation predictions
    val_predictions = val_df[['game_id', 'season', 'week', 'home_team', 'away_team']].copy()
    val_predictions['y_true'] = y_val.values
    val_predictions['y_pred_proba'] = val_proba_cal
    val_predictions['y_pred'] = y_pred
    val_predictions['correct'] = (y_pred == y_val.values).astype(int)
    val_predictions.to_parquet(os.path.join(run_dir, "validation_predictions.parquet"))

    logger.info(f"\n{'='*60}")
    logger.info("TRAINING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Output directory: {run_dir}")
    logger.info(f"Final accuracy: {accuracy:.4f}")
    logger.info(f"Final Brier score: {brier:.4f}")

    return calibrated_model, config


def main():
    parser = argparse.ArgumentParser(
        description="Train final NFL prediction model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full training with walk-forward validation
  python scripts/train_final_model.py \\
      --features data/nfl/processed/game_features_phase3.parquet

  # Quick training without walk-forward
  python scripts/train_final_model.py \\
      --features data/nfl/processed/game_features_phase3.parquet \\
      --no-walk-forward

  # Train with specific seasons
  python scripts/train_final_model.py \\
      --features data/nfl/processed/game_features_phase3.parquet \\
      --train-start 2019 --train-end 2023 --val-season 2024
        """
    )

    parser.add_argument('--features', required=True, help='Path to features parquet')
    parser.add_argument('--output', default='models/artifacts/', help='Output directory')
    parser.add_argument('--train-start', type=int, default=2015, help='First training season')
    parser.add_argument('--train-end', type=int, default=2023, help='Last training season')
    parser.add_argument('--val-season', type=int, default=2024, help='Validation season')
    parser.add_argument('--model', default='gbm',
                       choices=['gbm', 'lr', 'ensemble', 'ft_transformer', 'tabnet'],
                       help='Model type')
    parser.add_argument('--calibration', default='isotonic',
                       choices=['platt', 'isotonic', 'temperature'],
                       help='Calibration method')
    parser.add_argument('--no-walk-forward', action='store_true',
                       help='Skip walk-forward validation')
    parser.add_argument('--no-qb-deviation', action='store_true',
                       help='Exclude QB deviation features')
    parser.add_argument('--walk-forward-start', type=int, default=None,
                       help='First test season for walk-forward')

    args = parser.parse_args()

    train_seasons = list(range(args.train_start, args.train_end + 1))

    model, config = train_final_model(
        features_path=args.features,
        output_dir=args.output,
        train_seasons=train_seasons,
        val_season=args.val_season,
        model_type=args.model,
        calibration_method=args.calibration,
        include_qb_deviation=not args.no_qb_deviation,
        run_walk_forward=not args.no_walk_forward,
        walk_forward_start_season=args.walk_forward_start,
    )

    print(f"\n{'='*60}")
    print("SUCCESS!")
    print(f"{'='*60}")
    print(f"Model saved to: {config.get('output_dir', args.output)}")
    print(f"Final accuracy: {config['accuracy']:.4f}")
    print(f"Final Brier score: {config['brier_score']:.4f}")


if __name__ == "__main__":
    main()
