"""
Phase 1 - Task 1.1: Train the Stacking Ensemble

This script trains all base models (Logistic Regression, XGBoost/GBM, FT-Transformer)
and then trains the stacking ensemble with logistic regression meta-learner.

Deliverables:
- Trained ensemble artifacts saved to models/artifacts/nfl_ensemble/
- Performance comparison table (accuracy, Brier score, ROI)
- Evaluation script comparing ensemble vs individual models
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import json
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.training.trainer import (
    run_advanced_training_pipeline,
    load_features,
    split_by_season,
    load_backtest_config,
    train_logistic_regression,
    train_gradient_boosting,
    train_ft_transformer,
    load_config,
)
from models.architectures.stacking_ensemble import StackingEnsemble
from models.architectures.logistic_regression import LogisticRegressionModel
from models.architectures.gradient_boosting import GradientBoostingModel
from models.architectures.ft_transformer import FTTransformerModel
from models.base import BaseModel
from eval.metrics import accuracy, brier_score, log_loss, calibration_buckets
from eval.backtest import calculate_roi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate_model_performance(
    model: BaseModel,
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    set_name: str,
) -> Dict:
    """
    Evaluate a model and return comprehensive metrics.
    
    Returns:
        Dictionary with accuracy, Brier score, log loss, calibration metrics, ROI
    """
    # Get predictions
    y_pred_proba = model.predict_proba(X)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    y_true = np.asarray(y)
    
    # Basic metrics
    acc = accuracy(y_true, y_pred)
    brier = brier_score(y_true, y_pred_proba)
    ll = log_loss(y_true, y_pred_proba)
    
    # Calibration metrics
    calib_df = calibration_buckets(y_true, y_pred_proba, n_bins=10)
    mean_calib_error = calib_df['calibration_error'].mean()
    
    # ROI calculation (if market data available)
    roi_results = {}
    if 'close_spread' in df.columns:
        # Calculate market implied probabilities from spread
        # Simple approximation: spread of -3 means ~60% win prob for home team
        market_probs = []
        for _, row in df.iterrows():
            spread = row.get('close_spread', 0)
            if pd.notna(spread):
                # Convert spread to probability (rough approximation)
                # Using logistic function: prob = 1 / (1 + exp(-spread/3))
                market_prob = 1 / (1 + np.exp(-spread / 3))
                market_probs.append(market_prob)
            else:
                market_probs.append(0.5)  # Default to 50% if no spread
        
        market_probs = np.array(market_probs)
        
        # Calculate ROI at different edge thresholds
        for edge_threshold in [0.03, 0.05]:
            roi_data = calculate_roi(
                y_true, y_pred_proba, market_probs,
                edge_threshold=edge_threshold,
                unit_bet_size=1.0,
                df=df,
            )
            roi_results[f'roi_threshold_{edge_threshold:.2f}'] = roi_data
    else:
        logger.warning(f"No market data (close_spread) available for ROI calculation on {set_name}")
    
    return {
        'set_name': set_name,
        'n_games': len(y_true),
        'accuracy': acc,
        'brier_score': brier,
        'log_loss': ll,
        'mean_calibration_error': mean_calib_error,
        'calibration_buckets': calib_df.to_dict('records'),
        'roi_results': roi_results,
    }


def compare_models(
    models: Dict[str, BaseModel],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    df_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    df_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_test: pd.DataFrame,
    output_dir: Path,
) -> Dict:
    """
    Compare all models and ensemble on train/val/test sets.
    
    Returns:
        Dictionary with comparison results
    """
    logger.info("\n" + "=" * 60)
    logger.info("Model Comparison")
    logger.info("=" * 60)
    
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"\nEvaluating {model_name}...")
        
        # Evaluate on all sets
        train_results = evaluate_model_performance(model, X_train, y_train, df_train, "train")
        val_results = evaluate_model_performance(model, X_val, y_val, df_val, "validation")
        test_results = evaluate_model_performance(model, X_test, y_test, df_test, "test")
        
        results[model_name] = {
            'train': train_results,
            'validation': val_results,
            'test': test_results,
        }
        
        # Log summary
        logger.info(f"  Train Accuracy: {train_results['accuracy']:.4f}")
        logger.info(f"  Val Accuracy: {val_results['accuracy']:.4f}")
        logger.info(f"  Test Accuracy: {test_results['accuracy']:.4f}")
        logger.info(f"  Test Brier: {test_results['brier_score']:.4f}")
        logger.info(f"  Test Log Loss: {test_results['log_loss']:.4f}")
    
    # Create comparison table
    comparison_table = []
    for model_name, model_results in results.items():
        test = model_results['test']
        comparison_table.append({
            'model': model_name,
            'accuracy': test['accuracy'],
            'brier_score': test['brier_score'],
            'log_loss': test['log_loss'],
            'mean_calibration_error': test['mean_calibration_error'],
        })
    
    comparison_df = pd.DataFrame(comparison_table)
    comparison_df = comparison_df.sort_values('accuracy', ascending=False)
    
    logger.info("\n" + "=" * 60)
    logger.info("Comparison Summary (Test Set)")
    logger.info("=" * 60)
    logger.info("\n" + comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_path = output_dir / "model_comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    comparison_csv_path = output_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_csv_path, index=False)
    logger.info(f"\nSaved comparison to {comparison_path} and {comparison_csv_path}")
    
    return results


def train_phase1_ensemble(
    config_path: Path = None,
    artifacts_dir: Path = None,
    feature_table: str = None,
) -> Tuple[StackingEnsemble, Dict[str, BaseModel], Dict]:
    """
    Train Phase 1 ensemble with all base models.
    
    Args:
        config_path: Path to ensemble config file
        artifacts_dir: Directory to save artifacts
        feature_table: Feature table name
    
    Returns:
        Tuple of (ensemble_model, base_models_dict, evaluation_results)
    """
    logger.info("=" * 60)
    logger.info("Phase 1 - Task 1.1: Training Stacking Ensemble")
    logger.info("=" * 60)
    
    # Set up paths
    project_root = Path(__file__).parent.parent
    
    if config_path is None:
        config_path = project_root / "config" / "models" / "nfl_stacked_ensemble_v2.yaml"
    
    if artifacts_dir is None:
        artifacts_dir = project_root / "models" / "artifacts" / "nfl_ensemble"
    
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    config = load_config(config_path)
    logger.info(f"Loaded config from: {config_path}")
    
    # Load data
    logger.info("\nLoading feature data...")
    backtest_config = load_backtest_config()
    
    if feature_table is None:
        feature_table = config.get('features', {}).get('feature_table', 'baseline')
    
    X, y, feature_cols, df = load_features(feature_table=feature_table)
    
    # Split data
    train_seasons = backtest_config['splits']['train_seasons']
    val_season = backtest_config['splits']['validation_season']
    test_season = backtest_config['splits']['test_season']
    
    X_train, y_train, X_val, y_val, X_test, y_test = split_by_season(
        X, y, df, train_seasons, val_season, test_season
    )
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Align dataframe indices
    df_train = df[df["season"].isin(train_seasons)].copy().reset_index(drop=True)
    df_val = df[df["season"] == val_season].copy().reset_index(drop=True)
    df_test = df[df["season"] == test_season].copy().reset_index(drop=True)
    
    # Reset indices to align
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    # Step 1: Train base models
    logger.info("\n" + "=" * 60)
    logger.info("Step 1: Training Base Models")
    logger.info("=" * 60)
    
    base_models = {}
    
    # Train Logistic Regression
    logger.info("\nTraining Logistic Regression...")
    logit_config = load_config(project_root / "config" / "models" / "nfl_baseline.yaml")
    logit_model = train_logistic_regression(X_train, y_train, logit_config, artifacts_dir)
    base_models['logistic_regression'] = logit_model
    logger.info("✓ Logistic Regression trained")
    
    # Train Gradient Boosting
    logger.info("\nTraining Gradient Boosting...")
    gbm_model = train_gradient_boosting(X_train, y_train, logit_config, artifacts_dir)
    base_models['gradient_boosting'] = gbm_model
    logger.info("✓ Gradient Boosting trained")
    
    # Train FT-Transformer (optional, can skip if causing issues)
    logger.info("\nTraining FT-Transformer...")
    try:
        ft_config = load_config(project_root / "config" / "models" / "nfl_ft_transformer.yaml")
        ft_model = train_ft_transformer(X_train, y_train, X_val, y_val, ft_config, artifacts_dir)
        base_models['ft_transformer'] = ft_model
        logger.info("✓ FT-Transformer trained")
    except Exception as e:
        logger.warning(f"FT-Transformer training failed: {e}")
        logger.warning("Continuing without FT-Transformer...")
    
    # Step 2: Train Stacking Ensemble
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Training Stacking Ensemble")
    logger.info("=" * 60)
    
    meta_cfg = config.get('meta_model', {})
    stack_cfg = config.get('stacking', {})
    
    ensemble = StackingEnsemble(
        base_models=base_models,
        meta_model_type=meta_cfg.get('type', 'logistic'),
        include_features=stack_cfg.get('include_features', False),
        feature_fraction=stack_cfg.get('feature_fraction', 0.0),
        mlp_hidden_dims=meta_cfg.get('mlp_hidden_dims', [16, 8]),
        mlp_dropout=meta_cfg.get('mlp_dropout', 0.1),
        mlp_epochs=meta_cfg.get('mlp_epochs', 50),
        mlp_learning_rate=meta_cfg.get('mlp_learning_rate', 1e-3),
        random_state=config.get('random_state', 42),
    )
    
    logger.info(f"Fitting ensemble with {len(base_models)} base models")
    ensemble.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    
    # Save ensemble
    ensemble_path = artifacts_dir / "ensemble.pkl"
    ensemble.save(ensemble_path)
    logger.info(f"✓ Ensemble saved to {ensemble_path}")
    
    # Log ensemble weights
    weights = ensemble.get_model_weights()
    if weights:
        logger.info("\nEnsemble weights:")
        for model_name, weight in weights.items():
            logger.info(f"  {model_name}: {weight:.4f}")
    
    # Step 3: Evaluate all models
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Evaluating Models")
    logger.info("=" * 60)
    
    # Add ensemble to models dict
    all_models = base_models.copy()
    all_models['ensemble'] = ensemble
    
    # Compare all models
    results = compare_models(
        all_models,
        X_train, y_train, df_train,
        X_val, y_val, df_val,
        X_test, y_test, df_test,
        artifacts_dir,
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("Phase 1 - Task 1.1 Complete!")
    logger.info("=" * 60)
    logger.info(f"\nResults saved to: {artifacts_dir}")
    logger.info(f"  - Ensemble: ensemble.pkl")
    logger.info(f"  - Comparison: model_comparison.json")
    
    return ensemble, base_models, results


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Phase 1 - Task 1.1: Train Stacking Ensemble"
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to ensemble config file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for artifacts'
    )
    parser.add_argument(
        '--feature-table',
        type=str,
        default=None,
        help='Feature table name (baseline, phase2, phase2b)'
    )
    
    args = parser.parse_args()
    
    config_path = Path(args.config) if args.config else None
    artifacts_dir = Path(args.output_dir) if args.output_dir else None
    
    train_phase1_ensemble(
        config_path=config_path,
        artifacts_dir=artifacts_dir,
        feature_table=args.feature_table,
    )


if __name__ == "__main__":
    main()

