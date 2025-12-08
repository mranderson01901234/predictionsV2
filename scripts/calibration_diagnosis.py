#!/usr/bin/env python3
"""
Calibration Diagnosis Script for NFL Prediction Model

Run this to identify if calibration is the problem.

Your test results show:
- Accuracy: 57.14% (not terrible)
- Log Loss: 1.304 (TERRIBLE - worse than random)
- Calibration Error: 0.211 (21% off on average)

A log loss > 0.693 means the probability estimates are WORSE than 
predicting 50% for every game. This suggests calibration is broken.

Usage:
    python3 scripts/calibration_diagnosis.py --feature-table baseline --test-weeks 1-13
    python3 scripts/calibration_diagnosis.py --model-path artifacts/models/nfl_stacked_ensemble_v2/ensemble_v1.pkl
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
from datetime import datetime
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss,
    roc_auc_score, precision_score, recall_score
)
from sklearn.calibration import calibration_curve

from models.training.trainer import load_features
from models.base import BaseModel
from models.calibration import CalibratedModel
from models.architectures.stacking_ensemble import StackingEnsemble
from models.architectures.gradient_boosting import GradientBoostingModel
from models.architectures.ft_transformer import FTTransformerModel
from models.architectures.tabnet import TabNetModel
from models.architectures.logistic_regression import LogisticRegressionModel

# Visualization (optional)
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Matplotlib not available, skipping plots")


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get feature columns (exclude metadata and target)."""
    exclude_cols = {
        'game_id', 'season', 'week', 'date', 'gameday', 'game_date',
        'home_team', 'away_team', 'home_score', 'away_score',
        'result', 'home_win', 'target', 'spread_line', 'total_line',
        'close_spread', 'close_total', 'open_spread', 'open_total',
        'home_moneyline', 'away_moneyline', 'winning_team', 'losing_team'
    }
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    return feature_cols


def load_model(model_path: Path = None) -> BaseModel:
    """Load the trained model."""
    if model_path is None:
        # Try common paths
        project_root = Path(__file__).parent.parent
        possible_paths = [
            project_root / "artifacts" / "models" / "nfl_stacked_ensemble_v2" / "ensemble_v1.pkl",
            project_root / "models" / "artifacts" / "nfl_stacked_ensemble_v2" / "ensemble_v1.pkl",
            project_root / "artifacts" / "models" / "nfl_stacked_ensemble" / "ensemble_v1.pkl",
            project_root / "models" / "artifacts" / "leak_test_2025" / "ensemble_v1.pkl",
        ]
        for path in possible_paths:
            if path.exists():
                model_path = path
                break
        else:
            raise FileNotFoundError(
                f"Could not find model. Tried:\n" + 
                "\n".join(f"  - {p}" for p in possible_paths)
            )
    
    print(f"Loading model from: {model_path}")
    
    # Try to load as StackingEnsemble first (most common)
    try:
        # Custom loader for base models
        def custom_base_model_loader(path):
            path = Path(path)
            if 'ft_transformer' in str(path):
                return FTTransformerModel.load(path)
            elif 'gbm' in str(path) or 'gradient' in str(path):
                return GradientBoostingModel.load(path)
            elif 'logistic' in str(path) or 'logit' in str(path):
                return LogisticRegressionModel.load(path)
            elif 'tabnet' in str(path):
                return TabNetModel.load(path)
            else:
                return BaseModel.load(path)
        
        model = StackingEnsemble.load(model_path, base_model_loader=custom_base_model_loader)
        print(f"   Loaded as StackingEnsemble")
        return model
    except Exception as e:
        # Fallback to generic BaseModel.load
        print(f"   Failed to load as StackingEnsemble: {e}")
        print(f"   Trying generic BaseModel.load...")
        model = BaseModel.load(model_path)
        return model


def get_raw_predictions(model: BaseModel, X: pd.DataFrame) -> np.ndarray:
    """
    Get raw (uncalibrated) predictions.
    
    This tries to access the base estimator before calibration.
    """
    # Method 1: CalibratedModel wrapper
    if isinstance(model, CalibratedModel):
        if hasattr(model, 'base_model'):
            base = model.base_model
            if hasattr(base, 'predict_proba'):
                return base.predict_proba(X)
    
    # Method 2: Check for predict_proba_raw method
    if hasattr(model, 'predict_proba_raw'):
        return model.predict_proba_raw(X)
    
    # Method 3: Stacking ensemble - get base model predictions
    if hasattr(model, 'base_models'):
        # Average predictions from base models
        preds = []
        for name, base_model in model.base_models.items():
            if hasattr(base_model, 'predict_proba'):
                preds.append(base_model.predict_proba(X))
        if preds:
            return np.mean(preds, axis=0)
    
    # Method 4: Check for _base_model (private attribute)
    if hasattr(model, '_base_model'):
        base = model._base_model
        if hasattr(base, 'predict_proba'):
            return base.predict_proba(X)
    
    # Fallback: Return None (can't get raw predictions)
    print("WARNING: Could not extract raw predictions from model")
    print(f"Model type: {type(model)}")
    print(f"Model attributes: {[a for a in dir(model) if not a.startswith('__')][:20]}")
    return None


def get_calibrated_predictions(model: BaseModel, X: pd.DataFrame) -> np.ndarray:
    """Get calibrated predictions from the full model."""
    probs = model.predict_proba(X)
    # Handle 2D array (return second column if binary)
    if probs.ndim == 2 and probs.shape[1] == 2:
        return probs[:, 1]
    return probs


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute all relevant metrics."""
    y_pred = (y_prob > 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'brier_score': brier_score_loss(y_true, y_prob),
        'log_loss': log_loss(y_true, y_prob),
        'auc_roc': roc_auc_score(y_true, y_prob),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
    }
    
    # Calibration error (mean absolute difference from perfect calibration)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    metrics['mean_calibration_error'] = np.mean(np.abs(prob_true - prob_pred))
    
    # Expected calibration error (weighted by bin size)
    bin_counts = np.histogram(y_prob, bins=10, range=(0, 1))[0]
    bin_weights = bin_counts / len(y_prob)
    # Pad if necessary
    if len(bin_weights) > len(prob_true):
        bin_weights = bin_weights[:len(prob_true)]
    elif len(bin_weights) < len(prob_true):
        bin_weights = np.pad(bin_weights, (0, len(prob_true) - len(bin_weights)))
    metrics['expected_calibration_error'] = np.sum(bin_weights * np.abs(prob_true - prob_pred))
    
    return metrics


def compute_naive_baseline(y_true: np.ndarray) -> dict:
    """Compute metrics for naive 50% baseline."""
    y_prob = np.full(len(y_true), 0.5)
    return compute_metrics(y_true, y_prob)


def compute_home_bias_baseline(y_true: np.ndarray, home_win_rate: float = 0.53) -> dict:
    """Compute metrics for home-field advantage baseline (~53% home wins)."""
    y_prob = np.full(len(y_true), home_win_rate)
    return compute_metrics(y_true, y_prob)


def plot_calibration_comparison(
    y_true: np.ndarray,
    raw_probs: np.ndarray,
    cal_probs: np.ndarray,
    output_path: str = None
):
    """Plot calibration curves comparing raw vs calibrated."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plots")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Calibration curves
    ax1 = axes[0]
    
    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect', linewidth=2)
    
    # Raw calibration
    if raw_probs is not None:
        prob_true_raw, prob_pred_raw = calibration_curve(y_true, raw_probs, n_bins=10)
        ax1.plot(prob_pred_raw, prob_true_raw, 's-', label='Raw (uncalibrated)', color='blue', linewidth=2)
    
    # Calibrated
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, cal_probs, n_bins=10)
    ax1.plot(prob_pred_cal, prob_true_cal, 'o-', label='Calibrated', color='red', linewidth=2)
    
    ax1.set_xlabel('Predicted Probability')
    ax1.set_ylabel('True Probability')
    ax1.set_title('Calibration Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Probability distributions
    ax2 = axes[1]
    
    if raw_probs is not None:
        ax2.hist(raw_probs, bins=20, alpha=0.5, label='Raw', color='blue', edgecolor='black')
    ax2.hist(cal_probs, bins=20, alpha=0.5, label='Calibrated', color='red', edgecolor='black')
    ax2.axvline(x=0.5, color='black', linestyle='--', label='50%', linewidth=2)
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Count')
    ax2.set_title('Probability Distributions')
    ax2.legend()
    
    # Plot 3: Reliability diagram (binned accuracy)
    ax3 = axes[2]
    
    def binned_accuracy(y_true, y_prob, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_accs = []
        bin_counts = []
        
        for i in range(n_bins):
            mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
            if mask.sum() > 0:
                bin_accs.append(y_true[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_accs.append(np.nan)
                bin_counts.append(0)
        
        return bin_centers, np.array(bin_accs), np.array(bin_counts)
    
    # Perfect line
    ax3.plot([0, 1], [0, 1], 'k--', label='Perfect', linewidth=2)
    
    if raw_probs is not None:
        centers, accs, counts = binned_accuracy(y_true, raw_probs)
        valid = ~np.isnan(accs)
        ax3.scatter(centers[valid], accs[valid], s=counts[valid]*2, alpha=0.6, label='Raw', color='blue')
    
    centers, accs, counts = binned_accuracy(y_true, cal_probs)
    valid = ~np.isnan(accs)
    ax3.scatter(centers[valid], accs[valid], s=counts[valid]*2, alpha=0.6, label='Calibrated', color='red')
    
    ax3.set_xlabel('Predicted Probability')
    ax3.set_ylabel('Actual Win Rate')
    ax3.set_title('Reliability Diagram (size = sample count)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved calibration plot to: {output_path}")
    
    plt.close()


def diagnose_calibration(
    feature_table: str = "baseline",
    model_path: Path = None,
    test_season: int = 2025,
    test_weeks: list = None,
    output_dir: str = "results/calibration_diagnosis"
):
    """
    Main diagnostic function.
    
    Compares raw vs calibrated predictions to identify if calibration is the problem.
    """
    if test_weeks is None:
        test_weeks = list(range(1, 14))
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CALIBRATION DIAGNOSIS")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading test data...")
    X, y, feature_cols, df = load_features(feature_table=feature_table)
    
    # Filter to test set
    test_mask = (df['season'] == test_season) & (df['week'].isin(test_weeks))
    X_test = X[test_mask].copy()
    y_test = y[test_mask].copy()
    df_test = df[test_mask].copy()
    
    print(f"   Loaded {len(X_test)} test games from {test_season} weeks {min(test_weeks)}-{max(test_weeks)}")
    
    # Load model
    print("\n2. Loading model...")
    model = load_model(model_path)
    
    print(f"   Using {len(feature_cols)} features")
    
    # Handle NaN in features
    if X_test.isnull().any().any():
        print(f"   WARNING: {X_test.isnull().sum().sum()} NaN values in features, filling with 0")
        X_test = X_test.fillna(0)
    
    # Get predictions
    print("\n3. Getting predictions...")
    
    print("   Getting calibrated predictions...")
    cal_probs = get_calibrated_predictions(model, X_test)
    
    print("   Getting raw (uncalibrated) predictions...")
    raw_probs_array = get_raw_predictions(model, X_test)
    if raw_probs_array is not None:
        if raw_probs_array.ndim == 2 and raw_probs_array.shape[1] == 2:
            raw_probs = raw_probs_array[:, 1]
        else:
            raw_probs = raw_probs_array
    else:
        raw_probs = None
    
    # Compute metrics
    print("\n4. Computing metrics...")
    
    results = {
        'test_season': test_season,
        'test_weeks': test_weeks,
        'n_games': len(y_test),
        'actual_home_win_rate': float(y_test.mean()),
    }
    
    # Naive baseline (50%)
    print("   Computing naive baseline (50%)...")
    naive_metrics = compute_naive_baseline(y_test)
    results['naive_baseline'] = naive_metrics
    
    # Home bias baseline (~53%)
    print("   Computing home bias baseline...")
    home_metrics = compute_home_bias_baseline(y_test, home_win_rate=y_test.mean())
    results['home_bias_baseline'] = home_metrics
    
    # Calibrated model
    print("   Computing calibrated model metrics...")
    cal_metrics = compute_metrics(y_test, cal_probs)
    results['calibrated'] = cal_metrics
    
    # Raw model (if available)
    if raw_probs is not None:
        print("   Computing raw model metrics...")
        raw_metrics = compute_metrics(y_test, raw_probs)
        results['raw'] = raw_metrics
    else:
        results['raw'] = None
    
    # Print comparison
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    
    print(f"\nTest Set: {test_season} Weeks {min(test_weeks)}-{max(test_weeks)} ({len(y_test)} games)")
    print(f"Actual Home Win Rate: {y_test.mean():.1%}")
    
    print("\n" + "-" * 60)
    print(f"{'Metric':<25} {'Naive':<12} {'Home Bias':<12} {'Raw':<12} {'Calibrated':<12}")
    print("-" * 60)
    
    metrics_to_show = ['accuracy', 'brier_score', 'log_loss', 'auc_roc', 'mean_calibration_error']
    
    for metric in metrics_to_show:
        naive_val = naive_metrics.get(metric, 'N/A')
        home_val = home_metrics.get(metric, 'N/A')
        raw_val = raw_metrics.get(metric, 'N/A') if results['raw'] else 'N/A'
        cal_val = cal_metrics.get(metric, 'N/A')
        
        # Format values
        def fmt(v):
            if v == 'N/A':
                return 'N/A'
            elif metric == 'accuracy':
                return f"{v:.1%}"
            else:
                return f"{v:.4f}"
        
        print(f"{metric:<25} {fmt(naive_val):<12} {fmt(home_val):<12} {fmt(raw_val):<12} {fmt(cal_val):<12}")
    
    print("-" * 60)
    
    # Diagnosis
    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    
    issues = []
    recommendations = []
    
    # Check if calibration helps or hurts
    if results['raw'] is not None:
        raw_ll = results['raw']['log_loss']
        cal_ll = results['calibrated']['log_loss']
        
        if cal_ll > raw_ll:
            issues.append(f"❌ CALIBRATION HURTS: Log loss increased from {raw_ll:.4f} to {cal_ll:.4f}")
            recommendations.append("→ Remove or retrain calibration layer")
        elif cal_ll > 0.693:
            issues.append(f"❌ CALIBRATION BROKEN: Log loss {cal_ll:.4f} > 0.693 (worse than random)")
            recommendations.append("→ Retrain calibration with more data or different method")
        else:
            print(f"✓ Calibration helps: Log loss improved from {raw_ll:.4f} to {cal_ll:.4f}")
        
        # Check accuracy
        raw_acc = results['raw']['accuracy']
        cal_acc = results['calibrated']['accuracy']
        
        if cal_acc < raw_acc - 0.01:
            issues.append(f"⚠️ Calibration reduces accuracy: {raw_acc:.1%} → {cal_acc:.1%}")
        
        # Check if raw model is actually good
        if raw_acc > 0.60:
            print(f"✓ Raw model accuracy is decent: {raw_acc:.1%}")
        elif raw_acc > 0.55:
            print(f"⚠️ Raw model accuracy is marginal: {raw_acc:.1%}")
            recommendations.append("→ Consider adding more features (NGS, injuries, rest)")
        else:
            issues.append(f"❌ Raw model accuracy is poor: {raw_acc:.1%}")
            recommendations.append("→ Need better features or different model architecture")
    
    # Check calibration error
    cal_error = results['calibrated']['mean_calibration_error']
    if cal_error > 0.15:
        issues.append(f"❌ Severe miscalibration: {cal_error:.1%} mean error")
        recommendations.append("→ Try isotonic regression or temperature scaling")
    elif cal_error > 0.10:
        issues.append(f"⚠️ Moderate miscalibration: {cal_error:.1%} mean error")
    
    # Check Brier score
    brier = results['calibrated']['brier_score']
    if brier > 0.25:
        issues.append(f"❌ Poor Brier score: {brier:.4f} (should be < 0.25)")
    
    # Check vs naive baseline
    cal_ll = results['calibrated']['log_loss']
    naive_ll = results['naive_baseline']['log_loss']
    
    if cal_ll > naive_ll:
        issues.append(f"❌ MODEL WORSE THAN NAIVE: Predicting 50% every time would be better!")
        recommendations.append("→ This is a critical failure - investigate immediately")
    
    # Print issues
    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✓ No major issues found")
    
    # Print recommendations
    if recommendations:
        print("\nRECOMMENDATIONS:")
        for rec in recommendations:
            print(f"  {rec}")
    
    # Primary recommendation
    print("\n" + "=" * 60)
    print("PRIMARY RECOMMENDATION")
    print("=" * 60)
    
    if results['raw'] is not None and results['raw']['log_loss'] < results['calibrated']['log_loss']:
        print("""
The CALIBRATION LAYER is making the model WORSE.

Immediate fix:
1. Use raw predictions instead of calibrated
2. Re-train calibration with:
   - More validation data (use 2023-2024 instead of just 2024)
   - Different method (try Platt scaling instead of isotonic)
   - Cross-validation during calibration

Code to use raw predictions:
```python
# Instead of:
probs = model.predict_proba(X)[:, 1]

# Use base estimator:
if isinstance(model, CalibratedModel):
    probs = model.base_model.predict_proba(X)[:, 1]
```
""")
    elif results['calibrated']['accuracy'] < 0.58:
        print("""
The MODEL ITSELF needs improvement (features are too weak).

Recommended additions:
1. NGS features (CPOE, time to throw, separation) - Expected +1-2%
2. Rest days differential - Expected +0.5-1%
3. Injury data with severity - Expected +1-2%
4. QB deviation from career average - Expected +0.5-1%

Use the NGS API scraper to get started.
""")
    else:
        print("""
Model performance is acceptable but calibration needs work.

Try:
1. Temperature scaling (simplest)
2. Platt scaling (better for small datasets)
3. Isotonic regression with cross-validation
""")
    
    # Save results
    results_path = output_dir / "calibration_diagnosis.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to: {results_path}")
    
    # Plot calibration curves
    if HAS_MATPLOTLIB:
        plot_path = output_dir / "calibration_comparison.png"
        plot_calibration_comparison(y_test.values, raw_probs, cal_probs, str(plot_path))
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose calibration issues")
    parser.add_argument("--feature-table", default="baseline", help="Feature table name")
    parser.add_argument("--model-path", default=None, help="Path to model file")
    parser.add_argument("--test-season", type=int, default=2025, help="Test season")
    parser.add_argument("--test-weeks", default="1-13", help="Test weeks (e.g., 1-13)")
    parser.add_argument("--output-dir", default="results/calibration_diagnosis", help="Output directory")
    
    args = parser.parse_args()
    
    # Parse weeks
    if "-" in args.test_weeks:
        start, end = map(int, args.test_weeks.split("-"))
        test_weeks = list(range(start, end + 1))
    else:
        test_weeks = [int(w) for w in args.test_weeks.split(",")]
    
    model_path = Path(args.model_path) if args.model_path else None
    
    results = diagnose_calibration(
        feature_table=args.feature_table,
        model_path=model_path,
        test_season=args.test_season,
        test_weeks=test_weeks,
        output_dir=args.output_dir
    )

