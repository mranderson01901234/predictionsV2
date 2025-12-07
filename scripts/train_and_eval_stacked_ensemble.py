"""
Train and evaluate stacked ensemble model, comparing to baseline GBM.

This script:
1. Trains the stacked ensemble using ft_transformer, tabnet, and gbm as base learners
2. Saves predictions to artifacts/models/nfl_stacked_ensemble/
3. Evaluates ensemble vs baseline GBM
4. Saves evaluation results
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.training.trainer import (
    run_advanced_training_pipeline,
    load_features,
    split_by_season,
    load_backtest_config,
)
from models.architectures.gradient_boosting import GradientBoostingModel
from eval.backtest import evaluate_model
from eval.metrics import accuracy, brier_score, log_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def save_predictions(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    set_name: str,
    output_dir: Path,
):
    """
    Save model predictions to parquet file.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: True labels
        df: Full dataframe (for game_id)
        set_name: Name of dataset (train, val, test)
        output_dir: Directory to save predictions
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get predictions
    p_pred = model.predict_proba(X)
    y_pred = (p_pred >= 0.5).astype(int)
    
    # Align indices
    if not X.index.equals(df.index):
        # Reset indices to align
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        df = df.reset_index(drop=True)
    
    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        'game_id': df['game_id'].values,
        'y_true': y.values,
        'y_pred': y_pred,
        'p_pred': p_pred,
    })
    
    # Save to parquet
    output_path = output_dir / f"predictions_{set_name}.parquet"
    predictions_df.to_parquet(output_path, index=False)
    logger.info(f"Saved {set_name} predictions to {output_path}")
    logger.info(f"  Games: {len(predictions_df)}")
    logger.info(f"  Accuracy: {accuracy(y.values, y_pred):.4f}")
    logger.info(f"  Brier Score: {brier_score(y.values, p_pred):.4f}")
    
    return predictions_df


def compare_models(
    ensemble_model,
    baseline_gbm,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    df_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_test: pd.DataFrame,
    output_dir: Path,
):
    """
    Compare ensemble model to baseline GBM.
    
    Args:
        ensemble_model: Trained ensemble model
        baseline_gbm: Baseline GBM model
        X_val, y_val, df_val: Validation set
        X_test, y_test, df_test: Test set
        output_dir: Directory to save comparison results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "=" * 60)
    logger.info("Model Comparison: Ensemble vs Baseline GBM")
    logger.info("=" * 60)
    
    # Evaluate both models
    edge_thresholds = [0.03, 0.05]
    
    logger.info("\nEvaluating Ensemble Model...")
    ensemble_val_results = evaluate_model(
        ensemble_model, X_val, y_val, df_val, "validation", edge_thresholds
    )
    ensemble_test_results = evaluate_model(
        ensemble_model, X_test, y_test, df_test, "test", edge_thresholds
    )
    
    logger.info("\nEvaluating Baseline GBM...")
    gbm_val_results = evaluate_model(
        baseline_gbm, X_val, y_val, df_val, "validation", edge_thresholds
    )
    gbm_test_results = evaluate_model(
        baseline_gbm, X_test, y_test, df_test, "test", edge_thresholds
    )
    
    # Create comparison summary
    comparison = {
        'validation': {
            'ensemble': {
                'accuracy': ensemble_val_results['accuracy'],
                'brier_score': ensemble_val_results['brier_score'],
                'log_loss': ensemble_val_results['log_loss'],
                'mean_calibration_error': ensemble_val_results['mean_calibration_error'],
                'roi_3pct': ensemble_val_results['roi_results'].get('roi_threshold_0.03', {}).get('roi', 0.0),
                'roi_5pct': ensemble_val_results['roi_results'].get('roi_threshold_0.05', {}).get('roi', 0.0),
            },
            'baseline_gbm': {
                'accuracy': gbm_val_results['accuracy'],
                'brier_score': gbm_val_results['brier_score'],
                'log_loss': gbm_val_results['log_loss'],
                'mean_calibration_error': gbm_val_results['mean_calibration_error'],
                'roi_3pct': gbm_val_results['roi_results'].get('roi_threshold_0.03', {}).get('roi', 0.0),
                'roi_5pct': gbm_val_results['roi_results'].get('roi_threshold_0.05', {}).get('roi', 0.0),
            },
        },
        'test': {
            'ensemble': {
                'accuracy': ensemble_test_results['accuracy'],
                'brier_score': ensemble_test_results['brier_score'],
                'log_loss': ensemble_test_results['log_loss'],
                'mean_calibration_error': ensemble_test_results['mean_calibration_error'],
                'roi_3pct': ensemble_test_results['roi_results'].get('roi_threshold_0.03', {}).get('roi', 0.0),
                'roi_5pct': ensemble_test_results['roi_results'].get('roi_threshold_0.05', {}).get('roi', 0.0),
            },
            'baseline_gbm': {
                'accuracy': gbm_test_results['accuracy'],
                'brier_score': gbm_test_results['brier_score'],
                'log_loss': gbm_test_results['log_loss'],
                'mean_calibration_error': gbm_test_results['mean_calibration_error'],
                'roi_3pct': gbm_test_results['roi_results'].get('roi_threshold_0.03', {}).get('roi', 0.0),
                'roi_5pct': gbm_test_results['roi_results'].get('roi_threshold_0.05', {}).get('roi', 0.0),
            },
        },
    }
    
    # Calculate improvements
    for split in ['validation', 'test']:
        ensemble = comparison[split]['ensemble']
        gbm = comparison[split]['baseline_gbm']
        
        comparison[split]['improvements'] = {
            'accuracy': ensemble['accuracy'] - gbm['accuracy'],
            'brier_score': gbm['brier_score'] - ensemble['brier_score'],  # Lower is better
            'log_loss': gbm['log_loss'] - ensemble['log_loss'],  # Lower is better
            'mean_calibration_error': gbm['mean_calibration_error'] - ensemble['mean_calibration_error'],  # Lower is better
            'roi_3pct': ensemble['roi_3pct'] - gbm['roi_3pct'],
            'roi_5pct': ensemble['roi_5pct'] - gbm['roi_5pct'],
        }
    
    # Save comparison
    comparison_path = output_dir / "comparison_vs_baseline_gbm.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"\nSaved comparison results to {comparison_path}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Comparison Summary")
    logger.info("=" * 60)
    
    logger.info("\nValidation Set:")
    logger.info(f"  Accuracy: Ensemble={comparison['validation']['ensemble']['accuracy']:.4f}, "
                f"GBM={comparison['validation']['baseline_gbm']['accuracy']:.4f}, "
                f"Δ={comparison['validation']['improvements']['accuracy']:+.4f}")
    logger.info(f"  Brier: Ensemble={comparison['validation']['ensemble']['brier_score']:.4f}, "
                f"GBM={comparison['validation']['baseline_gbm']['brier_score']:.4f}, "
                f"Δ={comparison['validation']['improvements']['brier_score']:+.4f}")
    logger.info(f"  Log Loss: Ensemble={comparison['validation']['ensemble']['log_loss']:.4f}, "
                f"GBM={comparison['validation']['baseline_gbm']['log_loss']:.4f}, "
                f"Δ={comparison['validation']['improvements']['log_loss']:+.4f}")
    logger.info(f"  ROI (3%): Ensemble={comparison['validation']['ensemble']['roi_3pct']:.2%}, "
                f"GBM={comparison['validation']['baseline_gbm']['roi_3pct']:.2%}, "
                f"Δ={comparison['validation']['improvements']['roi_3pct']:+.2%}")
    
    logger.info("\nTest Set:")
    logger.info(f"  Accuracy: Ensemble={comparison['test']['ensemble']['accuracy']:.4f}, "
                f"GBM={comparison['test']['baseline_gbm']['accuracy']:.4f}, "
                f"Δ={comparison['test']['improvements']['accuracy']:+.4f}")
    logger.info(f"  Brier: Ensemble={comparison['test']['ensemble']['brier_score']:.4f}, "
                f"GBM={comparison['test']['baseline_gbm']['brier_score']:.4f}, "
                f"Δ={comparison['test']['improvements']['brier_score']:+.4f}")
    logger.info(f"  Log Loss: Ensemble={comparison['test']['ensemble']['log_loss']:.4f}, "
                f"GBM={comparison['test']['baseline_gbm']['log_loss']:.4f}, "
                f"Δ={comparison['test']['improvements']['log_loss']:+.4f}")
    logger.info(f"  ROI (3%): Ensemble={comparison['test']['ensemble']['roi_3pct']:.2%}, "
                f"GBM={comparison['test']['baseline_gbm']['roi_3pct']:.2%}, "
                f"Δ={comparison['test']['improvements']['roi_3pct']:+.2%}")
    
    return comparison


def main():
    """Main training and evaluation pipeline."""
    logger.info("=" * 60)
    logger.info("Stacked Ensemble Training and Evaluation")
    logger.info("=" * 60)
    
    # Set up paths
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "models" / "nfl_stacked_ensemble.yaml"
    # Use artifacts/models/ as requested by user
    artifacts_dir = project_root / "artifacts" / "models" / "nfl_stacked_ensemble"
    baseline_gbm_path = project_root / "models" / "artifacts" / "nfl_baseline" / "gbm.pkl"
    
    # Step 1: Train ensemble
    logger.info("\n[Step 1/4] Training Stacked Ensemble...")
    ensemble_model, X_train, y_train, X_val, y_val, X_test, y_test = run_advanced_training_pipeline(
        model_type='stacking_ensemble',
        config_path=config_path,
        artifacts_dir=artifacts_dir,
        apply_calibration_flag=True,
    )
    logger.info("✓ Ensemble training complete")
    
    # Step 2: Load baseline GBM
    logger.info("\n[Step 2/4] Loading Baseline GBM...")
    if not baseline_gbm_path.exists():
        raise FileNotFoundError(
            f"Baseline GBM not found at {baseline_gbm_path}. "
            "Please train baseline models first."
        )
    baseline_gbm = GradientBoostingModel.load(baseline_gbm_path)
    logger.info("✓ Baseline GBM loaded")
    
    # Step 3: Load full dataframe for predictions
    logger.info("\n[Step 3/4] Loading full dataframe...")
    backtest_config = load_backtest_config()
    feature_table = backtest_config.get("feature_table", "baseline")
    X_full, y_full, feature_cols, df_full = load_features(feature_table=feature_table)
    
    # Split dataframe
    train_seasons = backtest_config['splits']['train_seasons']
    val_season = backtest_config['splits']['validation_season']
    test_season = backtest_config['splits']['test_season']
    
    df_train = df_full[df_full["season"].isin(train_seasons)].copy()
    df_val = df_full[df_full["season"] == val_season].copy()
    df_test = df_full[df_full["season"] == test_season].copy()
    
    # Align indices
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)
    
    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    logger.info("✓ Data loaded and aligned")
    
    # Step 4: Save predictions
    logger.info("\n[Step 4/4] Saving predictions...")
    save_predictions(ensemble_model, X_train, y_train, df_train, "train", artifacts_dir)
    save_predictions(ensemble_model, X_val, y_val, df_val, "val", artifacts_dir)
    save_predictions(ensemble_model, X_test, y_test, df_test, "test", artifacts_dir)
    logger.info("✓ Predictions saved")
    
    # Step 5: Compare to baseline GBM
    logger.info("\n[Step 5/5] Comparing to Baseline GBM...")
    comparison = compare_models(
        ensemble_model,
        baseline_gbm,
        X_val, y_val, df_val,
        X_test, y_test, df_test,
        artifacts_dir,
    )
    logger.info("✓ Comparison complete")
    
    logger.info("\n" + "=" * 60)
    logger.info("Training and Evaluation Complete!")
    logger.info("=" * 60)
    logger.info(f"\nResults saved to: {artifacts_dir}")
    logger.info(f"  - Model: ensemble_v1.pkl")
    logger.info(f"  - Predictions: predictions_train/val/test.parquet")
    logger.info(f"  - Comparison: comparison_vs_baseline_gbm.json")


if __name__ == "__main__":
    main()

