#!/usr/bin/env python3
"""
Test Raw Model Performance

Quick test to confirm raw (uncalibrated) predictions perform better
than calibrated predictions.

Usage:
    python3 scripts/test_raw_model.py --model-path models/artifacts/leak_test_2025/ensemble_v1.pkl
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.training.trainer import load_features
from models.raw_ensemble import RawEnsemble
from models.architectures.stacking_ensemble import StackingEnsemble
from models.architectures.gradient_boosting import GradientBoostingModel
from models.architectures.ft_transformer import FTTransformerModel
from models.architectures.logistic_regression import LogisticRegressionModel
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score


def load_model(model_path: Path):
    """Load stacking ensemble model."""
    def custom_base_model_loader(path):
        path = Path(path)
        if 'ft_transformer' in str(path):
            return FTTransformerModel.load(path)
        elif 'gbm' in str(path) or 'gradient' in str(path):
            return GradientBoostingModel.load(path)
        elif 'logistic' in str(path) or 'logit' in str(path):
            return LogisticRegressionModel.load(path)
        else:
            from models.base import BaseModel
            return BaseModel.load(path)
    
    return StackingEnsemble.load(model_path, base_model_loader=custom_base_model_loader)


def compute_metrics(y_true, y_prob):
    """Compute metrics."""
    y_pred = (y_prob > 0.5).astype(int)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'log_loss': log_loss(y_true, y_prob),
        'brier_score': brier_score_loss(y_true, y_prob),
        'auc_roc': roc_auc_score(y_true, y_prob),
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test raw model performance")
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/artifacts/leak_test_2025/ensemble_v1.pkl',
        help='Path to model'
    )
    parser.add_argument(
        '--feature-table',
        type=str,
        default='baseline',
        help='Feature table name'
    )
    parser.add_argument(
        '--test-season',
        type=int,
        default=2025,
        help='Test season'
    )
    parser.add_argument(
        '--test-weeks',
        type=str,
        default='1-13',
        help='Test weeks'
    )
    
    args = parser.parse_args()
    
    # Parse weeks
    if '-' in args.test_weeks:
        start, end = map(int, args.test_weeks.split('-'))
        test_weeks = list(range(start, end + 1))
    else:
        test_weeks = [int(w) for w in args.test_weeks.split(',')]
    
    print("=" * 60)
    print("RAW MODEL PERFORMANCE TEST")
    print("=" * 60)
    
    # Load model
    print(f"\n1. Loading model from {args.model_path}...")
    model = load_model(Path(args.model_path))
    print(f"   ✓ Model loaded")
    print(f"   Base models: {list(model.base_models.keys())}")
    
    # Load features
    print(f"\n2. Loading features from table '{args.feature_table}'...")
    X, y, feature_cols, df = load_features(feature_table=args.feature_table)
    
    # Filter to test set
    test_mask = (df['season'] == args.test_season) & (df['week'].isin(test_weeks))
    X_test = X[test_mask].fillna(0)
    y_test = y[test_mask]
    
    print(f"   ✓ Loaded {len(X_test)} test games")
    print(f"   Actual home win rate: {y_test.mean():.1%}")
    
    # Get calibrated predictions
    print(f"\n3. Getting calibrated predictions...")
    cal_probs = model.predict_proba(X_test)
    if cal_probs.ndim == 2:
        cal_probs = cal_probs[:, 1]
    
    # Get raw predictions (individual base models)
    print(f"\n4. Getting raw predictions from base models...")
    raw_preds = {}
    for name, base_model in model.base_models.items():
        prob = base_model.predict_proba(X_test)
        if prob.ndim == 2:
            prob = prob[:, 1]
        raw_preds[name] = prob
        print(f"   {name}: {(prob > 0.5).mean():.1%} predicted home wins")
    
    # Average raw predictions
    raw_probs = np.mean(list(raw_preds.values()), axis=0)
    
    # Create RawEnsemble wrapper
    print(f"\n5. Creating RawEnsemble wrapper...")
    raw_model = RawEnsemble(model)
    wrapper_probs = raw_model.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    print(f"\n6. Computing metrics...")
    
    cal_metrics = compute_metrics(y_test.values, cal_probs)
    raw_metrics = compute_metrics(y_test.values, raw_probs)
    wrapper_metrics = compute_metrics(y_test.values, wrapper_probs)
    
    # Print comparison
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    
    print(f"\nTest Set: {args.test_season} Weeks {min(test_weeks)}-{max(test_weeks)} ({len(y_test)} games)")
    
    print("\n" + "-" * 60)
    print(f"{'Metric':<20} {'Calibrated':<15} {'Raw (Manual)':<15} {'Raw (Wrapper)':<15}")
    print("-" * 60)
    
    for metric in ['accuracy', 'log_loss', 'brier_score', 'auc_roc']:
        cal_val = cal_metrics[metric]
        raw_val = raw_metrics[metric]
        wrap_val = wrapper_metrics[metric]
        
        if metric == 'accuracy':
            fmt = lambda v: f"{v:.1%}"
        else:
            fmt = lambda v: f"{v:.4f}"
        
        print(f"{metric:<20} {fmt(cal_val):<15} {fmt(raw_val):<15} {fmt(wrap_val):<15}")
    
    print("-" * 60)
    
    # Improvement summary
    print("\n" + "=" * 60)
    print("IMPROVEMENT SUMMARY")
    print("=" * 60)
    
    acc_improvement = raw_metrics['accuracy'] - cal_metrics['accuracy']
    ll_improvement = cal_metrics['log_loss'] - raw_metrics['log_loss']  # Lower is better
    brier_improvement = cal_metrics['brier_score'] - raw_metrics['brier_score']  # Lower is better
    
    print(f"\nRaw vs Calibrated:")
    print(f"  Accuracy: {acc_improvement:+.1%} improvement")
    print(f"  Log Loss: {ll_improvement:+.4f} improvement (lower is better)")
    print(f"  Brier Score: {brier_improvement:+.4f} improvement (lower is better)")
    
    if raw_metrics['log_loss'] < cal_metrics['log_loss']:
        print(f"\n✓ RAW MODEL IS BETTER - Use RawEnsemble wrapper!")
    else:
        print(f"\n⚠️ Results are mixed - review metrics above")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print("""
Use RawEnsemble wrapper for production:

```python
from models.raw_ensemble import RawEnsemble

# Load model
model = StackingEnsemble.load('path/to/ensemble.pkl')

# Wrap to bypass calibration
raw_model = RawEnsemble(model)

# Use like normal model
probs = raw_model.predict_proba(X)[:, 1]
predictions = raw_model.predict(X)
```
""")


if __name__ == "__main__":
    main()

