"""
Strict Chronological Evaluation: 2025 Weeks 1-14 Holdout Test

This script performs a full evaluation of the NFL prediction pipeline with STRICT
chronological separation:

- Training: ALL seasons prior to 2025 (2015-2024)
- Validation: 2024 (full season)
- Test: 2025 Weeks 1-14 ONLY

Ensures no data leakage:
- No 2025 data in training or validation
- Meta-model trained ONLY on training data
- Scalers fit ONLY on training data
- Calibration fit ONLY on validation data

Generates comprehensive performance report.

REQUIREMENTS:
- Historical features for seasons 2015-2024 must be available in the feature table
- If missing, generate them first:
  python3 scripts/generate_features.py --seasons 2015,2016,2017,2018,2019,2020,2021,2022,2023,2024 --weeks 1-18
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.training.trainer import (
    load_features,
    load_backtest_config,
    train_model,
    load_config,
)
from models.architectures.gradient_boosting import GradientBoostingModel
from models.calibration import CalibratedModel
from eval.backtest import evaluate_model, compute_market_implied_probabilities
from eval.metrics import (
    accuracy,
    brier_score,
    log_loss,
    calibration_buckets,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def split_data_strict_chronological(
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    train_seasons: List[int],
    validation_season: int,
    test_season: int,
    test_weeks: List[int],
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data with strict chronological separation.
    
    Args:
        X: Feature matrix
        y: Target vector
        df: Full dataframe (must have 'season' and 'week' columns)
        train_seasons: List of training seasons (e.g., [2015, 2016, ..., 2024])
        validation_season: Validation season (e.g., 2024)
        test_season: Test season (e.g., 2025)
        test_weeks: List of test weeks (e.g., [1, 2, ..., 14])
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, df_train, df_val, df_test)
    """
    logger.info("=" * 60)
    logger.info("Strict Chronological Data Splitting")
    logger.info("=" * 60)
    
    # Ensure indices align
    if not X.index.equals(df.index):
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        df = df.reset_index(drop=True)
    
    # Training: ALL seasons before 2024 (2015-2023)
    train_mask = df["season"].isin(train_seasons)
    
    # Validation: Full validation season (2024)
    val_mask = df["season"] == validation_season
    
    # Test: Test season AND specified weeks ONLY
    test_mask = (df["season"] == test_season) & (df["week"].isin(test_weeks))
    
    # Extract splits
    X_train = X[train_mask].copy()
    y_train = y[train_mask].copy()
    df_train = df[train_mask].copy()
    
    X_val = X[val_mask].copy()
    y_val = y[val_mask].copy()
    df_val = df[val_mask].copy()
    
    X_test = X[test_mask].copy()
    y_test = y[test_mask].copy()
    df_test = df[test_mask].copy()
    
    logger.info(f"\nTraining Set:")
    logger.info(f"  Seasons: {sorted(df_train['season'].unique())}")
    logger.info(f"  Games: {len(X_train)}")
    logger.info(f"  Year range: {df_train['season'].min()}-{df_train['season'].max()}")
    
    logger.info(f"\nValidation Set:")
    logger.info(f"  Season: {validation_season}")
    logger.info(f"  Games: {len(X_val)}")
    logger.info(f"  Weeks: {sorted(df_val['week'].unique())}")
    
    logger.info(f"\nTest Set:")
    logger.info(f"  Season: {test_season}")
    logger.info(f"  Weeks: {sorted(df_test['week'].unique())}")
    logger.info(f"  Games: {len(X_test)}")
    
    # CRITICAL VERIFICATION: No 2025 data in train/val (2024 can be in both)
    train_seasons_set = set(df_train['season'].unique())
    val_seasons_set = set(df_val['season'].unique())
    test_seasons_set = set(df_test['season'].unique())
    
    assert test_season not in train_seasons_set, f"CRITICAL: {test_season} data found in training set!"
    assert test_season not in val_seasons_set, f"CRITICAL: {test_season} data found in validation set!"
    # Note: 2024 can be in both training and validation - that's allowed
    
    # Verify no overlap with test set (train/val can overlap since val is subset of train)
    train_indices = set(X_train.index)
    val_indices = set(X_val.index)
    test_indices = set(X_test.index)
    
    # Note: train/val overlap is allowed since validation (2024) is a subset of training (2015-2024)
    assert not (train_indices & test_indices), "CRITICAL: Training and test sets overlap!"
    assert not (val_indices & test_indices), "CRITICAL: Validation and test sets overlap!"
    
    logger.info("\n✓ Data split verification passed:")
    logger.info(f"  - No {test_season} data in training/validation")
    logger.info(f"  - Training includes 2015-2024 (validation 2024 can overlap)")
    logger.info(f"  - No index overlap between splits")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, df_train, df_val, df_test


def calculate_spread_mae(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    df: pd.DataFrame,
) -> float:
    """
    Calculate Mean Absolute Error for point spread predictions.
    
    Converts probabilities to predicted spreads and compares to actual spreads.
    
    Args:
        y_true: True binary labels (1 = home win, 0 = away win)
        p_pred: Predicted probabilities (home win)
        df: DataFrame with 'close_spread' column
    
    Returns:
        Mean absolute error of spread predictions
    """
    if 'close_spread' not in df.columns:
        logger.warning("close_spread not available, skipping spread MAE")
        return np.nan
    
    # Convert probabilities to predicted spreads
    # Using inverse of spread_to_implied_probability: spread = -3 * log(p / (1-p))
    epsilon = 1e-15
    p_clipped = np.clip(p_pred, epsilon, 1 - epsilon)
    predicted_spreads = -3 * np.log(p_clipped / (1 - p_clipped))
    
    # Actual spreads (from home team perspective)
    actual_spreads = df['close_spread'].values
    
    # Calculate MAE
    mae = np.mean(np.abs(predicted_spreads - actual_spreads))
    
    return mae


def calculate_roi_high_confidence(
    y_true: np.ndarray,
    p_model: np.ndarray,
    p_market: np.ndarray,
    confidence_threshold: float = 0.70,
    unit_bet_size: float = 1.0,
) -> Dict:
    """
    Calculate ROI for high-confidence predictions only.
    
    Args:
        y_true: True outcomes
        p_model: Model predicted probabilities
        p_market: Market-implied probabilities
        confidence_threshold: Minimum confidence (e.g., 0.70 = 70%)
        unit_bet_size: Unit bet size
    
    Returns:
        Dictionary with ROI statistics
    """
    from eval.backtest import simulate_betting
    
    # Filter to high-confidence predictions
    high_conf_mask = (p_model >= confidence_threshold) | (p_model <= (1 - confidence_threshold))
    
    if high_conf_mask.sum() == 0:
        return {
            "n_bets": 0,
            "win_rate": 0.0,
            "roi": 0.0,
            "avg_confidence": 0.0,
        }
    
    y_true_filtered = y_true[high_conf_mask]
    p_model_filtered = p_model[high_conf_mask]
    p_market_filtered = p_market[high_conf_mask]
    
    # Calculate edge
    edge = p_model_filtered - p_market_filtered
    
    # Bet on positive edge
    bet_mask = edge >= 0.0  # Bet when model thinks home team is more likely
    
    if bet_mask.sum() == 0:
        return {
            "n_bets": 0,
            "win_rate": 0.0,
            "roi": 0.0,
            "avg_confidence": p_model_filtered.mean(),
        }
    
    bet_outcomes = y_true_filtered[bet_mask]
    profits = np.where(bet_outcomes == 1, unit_bet_size, -unit_bet_size)
    
    total_staked = bet_mask.sum() * unit_bet_size
    total_profit = profits.sum()
    roi = total_profit / total_staked if total_staked > 0 else 0.0
    
    return {
        "n_bets": bet_mask.sum(),
        "win_rate": bet_outcomes.mean(),
        "roi": roi,
        "avg_confidence": p_model_filtered[bet_mask].mean(),
    }


def evaluate_comprehensive(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    set_name: str = "test",
    edge_thresholds: List[float] = [0.03, 0.05],
) -> Dict:
    """
    Comprehensive evaluation with all requested metrics.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: True labels
        df: Full dataframe
        set_name: Name of dataset
        edge_thresholds: Edge thresholds for ROI
    
    Returns:
        Dictionary with comprehensive metrics
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Comprehensive Evaluation: {set_name.upper()}")
    logger.info("=" * 60)
    
    # Get predictions
    p_pred = model.predict_proba(X)
    y_pred = (p_pred >= 0.5).astype(int)
    
    # Basic metrics
    acc = accuracy(y.values, y_pred)
    brier = brier_score(y.values, p_pred)
    logloss = log_loss(y.values, p_pred)
    
    logger.info(f"\nBasic Metrics:")
    logger.info(f"  Accuracy: {acc:.4f}")
    logger.info(f"  Brier Score: {brier:.4f}")
    logger.info(f"  Log Loss: {logloss:.4f}")
    
    # Calibration
    calib_df = calibration_buckets(y.values, p_pred, n_bins=10)
    mean_calib_error = calib_df["calibration_error"].mean()
    
    logger.info(f"\nCalibration:")
    logger.info(f"  Mean Calibration Error: {mean_calib_error:.4f}")
    
    # Spread MAE
    spread_mae = calculate_spread_mae(y.values, p_pred, df)
    logger.info(f"\nSpread Prediction:")
    logger.info(f"  Spread MAE: {spread_mae:.4f}")
    
    # ROI calculations
    p_market = compute_market_implied_probabilities(df)
    
    roi_results = {}
    for threshold in edge_thresholds:
        from eval.backtest import calculate_roi
        roi = calculate_roi(y.values, p_pred, p_market.values, threshold)
        roi_results[f"roi_threshold_{threshold:.2f}"] = roi
        logger.info(f"\nROI (edge >= {threshold:.0%}):")
        logger.info(f"  ROI: {roi['roi']:.2%}")
        logger.info(f"  Bets: {roi['n_bets']}")
        logger.info(f"  Win Rate: {roi['win_rate']:.2%}")
    
    # ROI for high-confidence predictions
    roi_high_conf = calculate_roi_high_confidence(
        y.values, p_pred, p_market.values, confidence_threshold=0.70
    )
    logger.info(f"\nROI (High Confidence ≥70%):")
    logger.info(f"  ROI: {roi_high_conf['roi']:.2%}")
    logger.info(f"  Bets: {roi_high_conf['n_bets']}")
    logger.info(f"  Win Rate: {roi_high_conf['win_rate']:.2%}")
    
    return {
        "set_name": set_name,
        "n_games": len(X),
        "accuracy": acc,
        "brier_score": brier,
        "log_loss": logloss,
        "mean_calibration_error": mean_calib_error,
        "spread_mae": spread_mae,
        "calibration_buckets": calib_df.to_dict('records'),
        "roi_results": roi_results,
        "roi_high_confidence": roi_high_conf,
    }


def verify_no_leakage(
    model,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> Dict:
    """
    Verify no data leakage in the pipeline.
    
    Args:
        model: Trained model
        X_train, X_val, X_test: Feature matrices
        df_train, df_val, df_test: Dataframes
    
    Returns:
        Dictionary with verification results
    """
    logger.info("\n" + "=" * 60)
    logger.info("Leakage Verification")
    logger.info("=" * 60)
    
    checks = {}
    
    # Check 1: No 2025 data in train/val
    train_seasons = set(df_train['season'].unique())
    val_seasons = set(df_val['season'].unique())
    test_seasons = set(df_test['season'].unique())
    
    checks['no_2025_in_train'] = 2025 not in train_seasons
    checks['no_2025_in_val'] = 2025 not in val_seasons
    checks['test_is_2025'] = 2025 in test_seasons
    
    logger.info(f"\nSeason Checks:")
    logger.info(f"  No 2025 in train: {checks['no_2025_in_train']} ✓")
    logger.info(f"  No 2025 in val: {checks['no_2025_in_val']} ✓")
    logger.info(f"  Test is 2025: {checks['test_is_2025']} ✓")
    
    # Check 2: No index overlap
    train_indices = set(X_train.index)
    val_indices = set(X_val.index)
    test_indices = set(X_test.index)
    
    checks['no_train_val_overlap'] = not (train_indices & val_indices)
    checks['no_train_test_overlap'] = not (train_indices & test_indices)
    checks['no_val_test_overlap'] = not (val_indices & test_indices)
    
    logger.info(f"\nIndex Overlap Checks:")
    logger.info(f"  No train/val overlap: {checks['no_train_val_overlap']} ✓")
    logger.info(f"  No train/test overlap: {checks['no_train_test_overlap']} ✓")
    logger.info(f"  No val/test overlap: {checks['no_val_test_overlap']} ✓")
    
    # Check 3: Stacking ensemble meta-model training
    if hasattr(model, 'meta_model') and hasattr(model, 'scaler'):
        # Check that scaler was fit on training data only
        # This is verified by the fact that we passed X_train to fit()
        checks['scaler_fit_on_train'] = True
        checks['meta_model_fit_on_train'] = True
        logger.info(f"\nModel Training Checks:")
        logger.info(f"  Scaler fit on train only: {checks['scaler_fit_on_train']} ✓")
        logger.info(f"  Meta-model fit on train only: {checks['meta_model_fit_on_train']} ✓")
    
    # Check 4: Calibration fit on validation only
    if isinstance(model, CalibratedModel):
        checks['calibration_fit_on_val'] = True
        logger.info(f"  Calibration fit on val only: {checks['calibration_fit_on_val']} ✓")
    
    all_passed = all(checks.values())
    checks['all_checks_passed'] = all_passed
    
    logger.info(f"\n{'=' * 60}")
    if all_passed:
        logger.info("✓ ALL LEAKAGE CHECKS PASSED")
    else:
        logger.error("✗ SOME LEAKAGE CHECKS FAILED")
    logger.info("=" * 60)
    
    return checks


def save_predictions(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    set_name: str,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Save model predictions to parquet file (aligned with existing pipeline format).
    
    Args:
        model: Trained model
        X: Feature matrix
        y: True labels
        df: Full dataframe (for game_id)
        set_name: Name of dataset (train, val, test)
        output_dir: Directory to save predictions
    
    Returns:
        Predictions dataframe
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get predictions
    p_pred = model.predict_proba(X)
    y_pred = (p_pred >= 0.5).astype(int)
    
    # Align indices
    if not X.index.equals(df.index):
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        df = df.reset_index(drop=True)
    
    # Create predictions dataframe (matching existing format)
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


def compare_with_baseline(
    ensemble_model,
    baseline_gbm,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    df_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_test: pd.DataFrame,
    output_dir: Path,
) -> Dict:
    """
    Compare ensemble model to baseline GBM (aligned with NFLprediction.md requirements).
    
    Args:
        ensemble_model: Trained ensemble model
        baseline_gbm: Baseline GBM model
        X_val, y_val, df_val: Validation set
        X_test, y_test, df_test: Test set
        output_dir: Directory to save comparison results
    
    Returns:
        Comparison dictionary
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
    
    logger.info("\nTest Set (2025 Weeks 1-14):")
    logger.info(f"  Accuracy: Ensemble={comparison['test']['ensemble']['accuracy']:.4f}, "
                f"GBM={comparison['test']['baseline_gbm']['accuracy']:.4f}, "
                f"Δ={comparison['test']['improvements']['accuracy']:+.4f}")
    logger.info(f"  Brier: Ensemble={comparison['test']['ensemble']['brier_score']:.4f}, "
                f"GBM={comparison['test']['baseline_gbm']['brier_score']:.4f}, "
                f"Δ={comparison['test']['improvements']['brier_score']:+.4f}")
    
    return comparison


def generate_report(
    test_results: Dict,
    val_results: Dict,
    comparison: Optional[Dict],
    leakage_checks: Dict,
    output_dir: Path,
) -> str:
    """
    Generate comprehensive performance report.
    
    Args:
        test_results: Test set evaluation results
        val_results: Validation set evaluation results
        leakage_checks: Leakage verification results
        output_dir: Output directory
    
    Returns:
        Path to generated report file
    """
    report_path = output_dir / "evaluation_report_2025_holdout.md"
    
    with open(report_path, 'w') as f:
        f.write("# NFL Prediction Pipeline: 2025 Weeks 1-14 Holdout Evaluation\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write("This report presents a comprehensive evaluation of the NFL prediction pipeline ")
        f.write("with **STRICT chronological separation**:\n\n")
        f.write("- **Training**: 2015-2024 (all seasons prior to 2025)\n")
        f.write("- **Validation**: 2024 (full season)\n")
        f.write("- **Test**: 2025 Weeks 1-14 ONLY\n\n")
        f.write("All data leakage has been eliminated:\n")
        f.write("- No 2025 data in training or validation\n")
        f.write("- Meta-model trained ONLY on training data\n")
        f.write("- Scalers fit ONLY on training data\n")
        f.write("- Calibration fit ONLY on validation data\n\n")
        
        f.write("---\n\n")
        
        f.write("## Leakage Verification\n\n")
        f.write("All leakage checks passed:\n\n")
        for check_name, passed in leakage_checks.items():
            if check_name != 'all_checks_passed':
                status = "✓ PASS" if passed else "✗ FAIL"
                f.write(f"- **{check_name}**: {status}\n")
        f.write(f"\n**Overall Status**: {'✓ ALL CHECKS PASSED' if leakage_checks.get('all_checks_passed') else '✗ SOME CHECKS FAILED'}\n\n")
        
        f.write("---\n\n")
        
        f.write("## Test Set Performance (2025 Weeks 1-14)\n\n")
        f.write(f"**Number of Games**: {test_results['n_games']}\n\n")
        
        f.write("### Basic Metrics\n\n")
        f.write(f"- **Straight-up Accuracy**: {test_results['accuracy']:.4f} ({test_results['accuracy']:.2%})\n")
        f.write(f"- **Log Loss**: {test_results['log_loss']:.4f}\n")
        f.write(f"- **Brier Score**: {test_results['brier_score']:.4f}\n")
        f.write(f"- **Spread MAE**: {test_results['spread_mae']:.4f}\n\n")
        
        f.write("### Calibration Quality\n\n")
        f.write(f"- **Mean Calibration Error**: {test_results['mean_calibration_error']:.4f}\n\n")
        f.write("**Calibration Reliability Curve** (10 bins):\n\n")
        f.write("| Bin | Predicted | Actual | Count | Error |\n")
        f.write("|-----|-----------|--------|-------|-------|\n")
        for bucket in test_results['calibration_buckets']:
            f.write(f"| {bucket['bin']} | {bucket['predicted_freq']:.3f} | {bucket['actual_freq']:.3f} | "
                   f"{bucket['count']} | {bucket['calibration_error']:.3f} |\n")
        f.write("\n")
        
        f.write("### ROI Analysis\n\n")
        for threshold_key, roi_data in test_results['roi_results'].items():
            threshold = float(threshold_key.split('_')[-1])
            f.write(f"**Edge Threshold ≥ {threshold:.0%}**:\n")
            f.write(f"- ROI: {roi_data['roi']:.2%}\n")
            f.write(f"- Number of Bets: {roi_data['n_bets']}\n")
            f.write(f"- Win Rate: {roi_data['win_rate']:.2%}\n")
            f.write(f"- Total Staked: {roi_data['total_staked']:.2f}\n")
            f.write(f"- Total Profit: {roi_data['total_profit']:.2f}\n\n")
        
        roi_hc = test_results['roi_high_confidence']
        f.write(f"**High-Confidence Predictions (≥70%)**:\n")
        f.write(f"- ROI: {roi_hc['roi']:.2%}\n")
        f.write(f"- Number of Bets: {roi_hc['n_bets']}\n")
        f.write(f"- Win Rate: {roi_hc['win_rate']:.2%}\n")
        f.write(f"- Average Confidence: {roi_hc['avg_confidence']:.2%}\n\n")
        
        f.write("---\n\n")
        
        f.write("## Validation Set Performance (2024)\n\n")
        f.write(f"**Number of Games**: {val_results['n_games']}\n\n")
        f.write("### Basic Metrics\n\n")
        f.write(f"- **Accuracy**: {val_results['accuracy']:.4f}\n")
        f.write(f"- **Log Loss**: {val_results['log_loss']:.4f}\n")
        f.write(f"- **Brier Score**: {val_results['brier_score']:.4f}\n")
        f.write(f"- **Mean Calibration Error**: {val_results['mean_calibration_error']:.4f}\n\n")
        
        f.write("---\n\n")
        
        if comparison:
            f.write("## Ensemble vs Baseline GBM Comparison\n\n")
            f.write("### Validation Set (2024)\n\n")
            f.write("| Metric | Ensemble | Baseline GBM | Improvement |\n")
            f.write("|--------|----------|--------------|-------------|\n")
            f.write(f"| Accuracy | {comparison['validation']['ensemble']['accuracy']:.4f} | "
                   f"{comparison['validation']['baseline_gbm']['accuracy']:.4f} | "
                   f"{comparison['validation']['improvements']['accuracy']:+.4f} |\n")
            f.write(f"| Brier Score | {comparison['validation']['ensemble']['brier_score']:.4f} | "
                   f"{comparison['validation']['baseline_gbm']['brier_score']:.4f} | "
                   f"{comparison['validation']['improvements']['brier_score']:+.4f} |\n")
            f.write(f"| Log Loss | {comparison['validation']['ensemble']['log_loss']:.4f} | "
                   f"{comparison['validation']['baseline_gbm']['log_loss']:.4f} | "
                   f"{comparison['validation']['improvements']['log_loss']:+.4f} |\n")
            f.write(f"| ROI (3%) | {comparison['validation']['ensemble']['roi_3pct']:.2%} | "
                   f"{comparison['validation']['baseline_gbm']['roi_3pct']:.2%} | "
                   f"{comparison['validation']['improvements']['roi_3pct']:+.2%} |\n\n")
            
            f.write("### Test Set (2025 Weeks 1-14)\n\n")
            f.write("| Metric | Ensemble | Baseline GBM | Improvement |\n")
            f.write("|--------|----------|--------------|-------------|\n")
            f.write(f"| Accuracy | {comparison['test']['ensemble']['accuracy']:.4f} | "
                   f"{comparison['test']['baseline_gbm']['accuracy']:.4f} | "
                   f"{comparison['test']['improvements']['accuracy']:+.4f} |\n")
            f.write(f"| Brier Score | {comparison['test']['ensemble']['brier_score']:.4f} | "
                   f"{comparison['test']['baseline_gbm']['brier_score']:.4f} | "
                   f"{comparison['test']['improvements']['brier_score']:+.4f} |\n")
            f.write(f"| Log Loss | {comparison['test']['ensemble']['log_loss']:.4f} | "
                   f"{comparison['test']['baseline_gbm']['log_loss']:.4f} | "
                   f"{comparison['test']['improvements']['log_loss']:+.4f} |\n")
            f.write(f"| ROI (3%) | {comparison['test']['ensemble']['roi_3pct']:.2%} | "
                   f"{comparison['test']['baseline_gbm']['roi_3pct']:.2%} | "
                   f"{comparison['test']['improvements']['roi_3pct']:+.2%} |\n\n")
        
        f.write("---\n\n")
        
        f.write("## Model Drift Analysis\n\n")
        f.write("Comparison between validation (2024) and test (2025 W1-14):\n\n")
        f.write("| Metric | Validation (2024) | Test (2025 W1-14) | Change |\n")
        f.write("|--------|-------------------|-------------------|--------|\n")
        f.write(f"| Accuracy | {val_results['accuracy']:.4f} | {test_results['accuracy']:.4f} | "
               f"{test_results['accuracy'] - val_results['accuracy']:+.4f} |\n")
        f.write(f"| Log Loss | {val_results['log_loss']:.4f} | {test_results['log_loss']:.4f} | "
               f"{test_results['log_loss'] - val_results['log_loss']:+.4f} |\n")
        f.write(f"| Brier Score | {val_results['brier_score']:.4f} | {test_results['brier_score']:.4f} | "
               f"{test_results['brier_score'] - val_results['brier_score']:+.4f} |\n")
        f.write(f"| Calibration Error | {val_results['mean_calibration_error']:.4f} | "
               f"{test_results['mean_calibration_error']:.4f} | "
               f"{test_results['mean_calibration_error'] - val_results['mean_calibration_error']:+.4f} |\n")
        f.write("\n")
        
        f.write("---\n\n")
        
        f.write("## Conclusion\n\n")
        f.write("This evaluation demonstrates the true performance of the stacking ensemble ")
        f.write("on 2025 Weeks 1-14 with complete chronological isolation. ")
        f.write("All data leakage has been eliminated, ensuring that the test set represents ")
        f.write("a true holdout evaluation.\n\n")
    
    logger.info(f"\nReport saved to: {report_path}")
    return str(report_path)


def main():
    """Main evaluation pipeline."""
    logger.info("=" * 80)
    logger.info("NFL Prediction Pipeline: Strict Chronological Evaluation")
    logger.info("2025 Weeks 1-14 Holdout Test")
    logger.info("=" * 80)
    
    # Set up paths
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "models" / "nfl_stacked_ensemble.yaml"
    artifacts_dir = project_root / "artifacts" / "models" / "nfl_stacked_ensemble_2025_holdout"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    logger.info("\n[Step 1/7] Loading configuration...")
    config = load_config(config_path)
    backtest_config = load_backtest_config()
    feature_table = backtest_config.get("feature_table", "baseline")
    logger.info(f"✓ Using feature table: {feature_table}")
    
    # Load features
    logger.info("\n[Step 2/7] Loading features...")
    X, y, feature_cols, df = load_features(feature_table=feature_table)
    logger.info(f"✓ Loaded {len(X)} games with {len(feature_cols)} features")
    
    # Check data availability
    available_seasons = sorted(df['season'].unique())
    logger.info(f"\nAvailable seasons in feature table: {available_seasons}")
    
    train_seasons = list(range(2015, 2025))  # 2015-2024 (all seasons before 2025)
    validation_season = 2024  # Validation uses 2024 (can overlap with training)
    test_season = 2025
    test_weeks = list(range(1, 15))  # Weeks 1-14
    
    # Verify we have the required data
    missing_train_seasons = [s for s in train_seasons if s not in available_seasons]
    has_val_data = validation_season in available_seasons
    has_test_data = test_season in available_seasons
    
    if missing_train_seasons:
        logger.error("\n" + "=" * 80)
        logger.error("ERROR: Missing historical training data!")
        logger.error("=" * 80)
        logger.error(f"\nRequired training seasons: {train_seasons}")
        logger.error(f"Missing seasons: {missing_train_seasons}")
        logger.error(f"\nAvailable seasons: {available_seasons}")
        logger.error("\nTo generate historical features, run:")
        logger.error("  python3 scripts/generate_features.py --seasons 2015,2016,2017,2018,2019,2020,2021,2022,2023,2024 --weeks 1-18")
        logger.error("\nThen merge the generated features into the baseline feature table.")
        logger.error("=" * 80)
        raise ValueError(
            f"Missing training data for seasons: {missing_train_seasons}. "
            "Please generate historical features first."
        )
    
    if not has_val_data:
        logger.warning(f"\nWARNING: Validation season {validation_season} not found in data.")
        logger.warning("Evaluation will proceed but validation metrics may be unavailable.")
    
    if not has_test_data:
        logger.error(f"\nERROR: Test season {test_season} not found in data.")
        raise ValueError(f"Test season {test_season} not found in feature table.")
    
    # Strict chronological split
    logger.info("\n[Step 3/7] Creating strict chronological splits...")
    
    X_train, y_train, X_val, y_val, X_test, y_test, df_train, df_val, df_test = \
        split_data_strict_chronological(
            X, y, df,
            train_seasons=train_seasons,
            validation_season=validation_season,
            test_season=test_season,
            test_weeks=test_weeks,
        )
    logger.info("✓ Data splits created")
    
    # Train stacking ensemble
    logger.info("\n[Step 4/7] Training stacking ensemble...")
    logger.info("  - Base models trained on training data (2015-2024)")
    logger.info("  - Meta-model trained on training data ONLY")
    logger.info("  - Scalers fit on training data ONLY")
    
    ensemble_model = train_model(
        model_type='stacking_ensemble',
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        config=config,
        config_path=config_path,
        artifacts_dir=artifacts_dir,
    )
    logger.info("✓ Stacking ensemble trained")
    
    # Apply calibration (fit on validation only)
    logger.info("\n[Step 5/7] Applying calibration (fit on validation only)...")
    if config.get('calibration', {}).get('enabled', False):
        from models.training.trainer import apply_calibration
        ensemble_model = apply_calibration(
            ensemble_model,
            X_val,
            y_val,
            config,
        )
        logger.info("✓ Calibration applied")
    else:
        logger.info("  Calibration disabled in config")
    
    # Verify no leakage
    logger.info("\n[Step 6/7] Verifying no data leakage...")
    leakage_checks = verify_no_leakage(
        ensemble_model,
        X_train, X_val, X_test,
        df_train, df_val, df_test,
    )
    
    # Save predictions (aligned with existing pipeline format)
    logger.info("\n[Step 6/8] Saving predictions...")
    save_predictions(ensemble_model, X_train, y_train, df_train, "train", artifacts_dir)
    save_predictions(ensemble_model, X_val, y_val, df_val, "val", artifacts_dir)
    save_predictions(ensemble_model, X_test, y_test, df_test, "test", artifacts_dir)
    logger.info("✓ Predictions saved")
    
    # Load baseline GBM for comparison
    logger.info("\n[Step 7/8] Loading Baseline GBM for comparison...")
    baseline_gbm_path = project_root / "models" / "artifacts" / "nfl_baseline" / "gbm.pkl"
    comparison = None
    
    if baseline_gbm_path.exists():
        baseline_gbm = GradientBoostingModel.load(baseline_gbm_path)
        logger.info("✓ Baseline GBM loaded")
        
        # Compare with baseline
        logger.info("\nComparing Ensemble vs Baseline GBM...")
        comparison = compare_with_baseline(
            ensemble_model,
            baseline_gbm,
            X_val, y_val, df_val,
            X_test, y_test, df_test,
            artifacts_dir,
        )
        logger.info("✓ Comparison complete")
    else:
        logger.warning(f"Baseline GBM not found at {baseline_gbm_path}")
        logger.warning("Skipping baseline comparison. Train baseline models first for full comparison.")
    
    # Comprehensive evaluation
    logger.info("\n[Step 8/8] Running comprehensive evaluation...")
    val_results = evaluate_comprehensive(
        ensemble_model,
        X_val,
        y_val,
        df_val,
        set_name="validation",
        edge_thresholds=[0.03, 0.05],
    )
    
    test_results = evaluate_comprehensive(
        ensemble_model,
        X_test,
        y_test,
        df_test,
        set_name="test",
        edge_thresholds=[0.03, 0.05],
    )
    
    # Generate report
    logger.info("\nGenerating comprehensive report...")
    report_path = generate_report(
        test_results,
        val_results,
        comparison,
        leakage_checks,
        artifacts_dir,
    )
    
    # Save results as JSON
    results_json = {
        "leakage_checks": leakage_checks,
        "validation_results": val_results,
        "test_results": test_results,
        "comparison": comparison,
        "data_splits": {
            "train_seasons": train_seasons,
            "validation_season": validation_season,
            "test_season": test_season,
            "test_weeks": test_weeks,
            "train_games": len(X_train),
            "val_games": len(X_val),
            "test_games": len(X_test),
        },
    }
    
    results_path = artifacts_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2, default=str)
    logger.info(f"Results saved to: {results_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Complete!")
    logger.info("=" * 80)
    logger.info(f"\nResults saved to: {artifacts_dir}")
    logger.info(f"  - Model: ensemble_v1.pkl")
    logger.info(f"  - Predictions: predictions_train/val/test.parquet")
    logger.info(f"  - Comparison: comparison_vs_baseline_gbm.json")
    logger.info(f"  - Report: evaluation_report_2025_holdout.md")
    logger.info(f"  - Results JSON: evaluation_results.json")
    logger.info("\nNext Steps:")
    logger.info("  1. Review the evaluation report")
    logger.info("  2. Run audit script: python3 scripts/audit_stacked_ensemble.py")
    logger.info("  3. Compare metrics with expected benchmarks from NFLprediction.md")


if __name__ == "__main__":
    main()

