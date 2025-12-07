"""
Audit script for stacked ensemble experiment results.

Checks for:
1. Evaluation metrics and logs
2. SHAP/feature importance files
3. Model artifacts and predictions
4. Comparison vs baseline models
"""

import sys
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_file_exists(file_path: Path, description: str) -> tuple[bool, Optional[str]]:
    """Check if file exists and return status."""
    if file_path.exists():
        try:
            size = file_path.stat().st_size
            return True, f"✓ Found ({size:,} bytes)"
        except Exception as e:
            return True, f"✓ Found (error reading: {e})"
    else:
        return False, "✗ Missing"


def load_json_safe(file_path: Path) -> Optional[dict]:
    """Safely load JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Error loading {file_path}: {e}")
        return None


def load_parquet_safe(file_path: Path) -> Optional[pd.DataFrame]:
    """Safely load parquet file."""
    try:
        return pd.read_parquet(file_path)
    except Exception as e:
        logger.warning(f"Error loading {file_path}: {e}")
        return None


def calculate_metrics_from_predictions(df: pd.DataFrame) -> Dict:
    """Calculate metrics from predictions dataframe."""
    if df is None or len(df) == 0:
        return {}
    
    y_true = df['y_true'].values
    y_pred = df['y_pred'].values if 'y_pred' in df.columns else None
    p_pred = df['p_pred'].values if 'p_pred' in df.columns else None
    
    metrics = {
        'n_samples': len(df),
    }
    
    if y_pred is not None:
        accuracy = (y_pred == y_true).mean()
        metrics['accuracy'] = accuracy
    
    if p_pred is not None:
        brier_score = ((p_pred - y_true) ** 2).mean()
        metrics['brier_score'] = brier_score
        
        # Log loss
        epsilon = 1e-15
        p_clipped = np.clip(p_pred, epsilon, 1 - epsilon)
        log_loss = -(y_true * np.log(p_clipped) + (1 - y_true) * np.log(1 - p_clipped)).mean()
        metrics['log_loss'] = log_loss
        
        # ROC-AUC (if sklearn available)
        try:
            from sklearn.metrics import roc_auc_score
            if len(np.unique(y_true)) > 1:
                roc_auc = roc_auc_score(y_true, p_pred)
                metrics['roc_auc'] = roc_auc
        except ImportError:
            pass
    
    return metrics


def audit_ensemble_results(artifacts_dir: Path) -> Dict:
    """
    Audit stacked ensemble experiment results.
    
    Args:
        artifacts_dir: Directory containing experiment artifacts
    
    Returns:
        Dictionary with audit results
    """
    artifacts_dir = Path(artifacts_dir)
    
    audit_results = {
        'directory': str(artifacts_dir),
        'exists': artifacts_dir.exists(),
        'files_found': [],
        'files_missing': [],
        'metrics': {},
        'comparison': {},
        'feature_importance': {},
        'warnings': [],
        'errors': [],
    }
    
    if not artifacts_dir.exists():
        audit_results['errors'].append(f"Directory does not exist: {artifacts_dir}")
        return audit_results
    
    logger.info(f"Auditing ensemble results in: {artifacts_dir}")
    
    # Expected files
    expected_files = {
        'model': artifacts_dir / "ensemble_v1.pkl",
        'predictions_train': artifacts_dir / "predictions_train.parquet",
        'predictions_val': artifacts_dir / "predictions_val.parquet",
        'predictions_test': artifacts_dir / "predictions_test.parquet",
        'comparison': artifacts_dir / "comparison_vs_baseline_gbm.json",
        'metrics': artifacts_dir / "metrics.json",
        'evaluation': artifacts_dir / "evaluation.parquet",
        'shap_values': artifacts_dir / "shap_values.npy",
        'feature_importance': artifacts_dir / "feature_importance.json",
        'calibration_model': artifacts_dir / "calibration_model.pkl",
    }
    
    # Check each file
    for file_type, file_path in expected_files.items():
        exists, status = check_file_exists(file_path, file_type)
        if exists:
            audit_results['files_found'].append({
                'type': file_type,
                'path': str(file_path),
                'status': status
            })
        else:
            audit_results['files_missing'].append({
                'type': file_type,
                'path': str(file_path),
                'status': status
            })
    
    # Load and analyze metrics
    if (artifacts_dir / "comparison_vs_baseline_gbm.json").exists():
        comparison_data = load_json_safe(artifacts_dir / "comparison_vs_baseline_gbm.json")
        if comparison_data:
            audit_results['comparison'] = comparison_data
    
    # Load predictions and calculate metrics
    for split in ['train', 'val', 'test']:
        pred_file = artifacts_dir / f"predictions_{split}.parquet"
        if pred_file.exists():
            df = load_parquet_safe(pred_file)
            if df is not None:
                metrics = calculate_metrics_from_predictions(df)
                audit_results['metrics'][split] = metrics
    
    # Check for feature importance
    feature_importance_file = artifacts_dir / "feature_importance.json"
    if feature_importance_file.exists():
        fi_data = load_json_safe(feature_importance_file)
        if fi_data:
            audit_results['feature_importance'] = fi_data
    
    # Check for SHAP values
    shap_file = artifacts_dir / "shap_values.npy"
    if shap_file.exists():
        try:
            shap_values = np.load(shap_file)
            audit_results['feature_importance']['shap_values_shape'] = str(shap_values.shape)
            audit_results['feature_importance']['shap_values_mean_abs'] = float(np.abs(shap_values).mean())
        except Exception as e:
            audit_results['warnings'].append(f"Error loading SHAP values: {e}")
    
    return audit_results


def print_audit_report(audit_results: Dict):
    """Print formatted audit report."""
    print("\n" + "=" * 80)
    print("STACKED ENSEMBLE EXPERIMENT AUDIT REPORT")
    print("=" * 80)
    
    print(f"\nDirectory: {audit_results['directory']}")
    print(f"Exists: {'Yes' if audit_results['exists'] else 'No'}")
    
    if not audit_results['exists']:
        print("\n❌ Directory does not exist. Training may not have been run yet.")
        return
    
    # Files found
    print(f"\n{'=' * 80}")
    print("FILES FOUND")
    print("=" * 80)
    if audit_results['files_found']:
        for file_info in audit_results['files_found']:
            print(f"  {file_info['type']:20s} {file_info['status']}")
    else:
        print("  No expected files found.")
    
    # Files missing
    print(f"\n{'=' * 80}")
    print("FILES MISSING")
    print("=" * 80)
    if audit_results['files_missing']:
        for file_info in audit_results['files_missing']:
            print(f"  {file_info['type']:20s} {file_info['status']}")
    else:
        print("  All expected files present!")
    
    # Metrics
    print(f"\n{'=' * 80}")
    print("EVALUATION METRICS")
    print("=" * 80)
    if audit_results['metrics']:
        for split, metrics in audit_results['metrics'].items():
            print(f"\n{split.upper()} Set:")
            print(f"  Samples: {metrics.get('n_samples', 'N/A')}")
            if 'accuracy' in metrics:
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
            if 'brier_score' in metrics:
                print(f"  Brier Score: {metrics['brier_score']:.4f}")
            if 'log_loss' in metrics:
                print(f"  Log Loss: {metrics['log_loss']:.4f}")
            if 'roc_auc' in metrics:
                print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    else:
        print("  No metrics found.")
    
    # Comparison vs Baseline
    print(f"\n{'=' * 80}")
    print("COMPARISON VS BASELINE GBM")
    print("=" * 80)
    if audit_results['comparison']:
        for split in ['validation', 'test']:
            if split in audit_results['comparison']:
                print(f"\n{split.upper()} Set:")
                split_data = audit_results['comparison'][split]
                
                if 'ensemble' in split_data and 'baseline_gbm' in split_data:
                    ensemble = split_data['ensemble']
                    gbm = split_data['baseline_gbm']
                    
                    print(f"\n  Ensemble:")
                    print(f"    Accuracy: {ensemble.get('accuracy', 'N/A'):.4f}" if isinstance(ensemble.get('accuracy'), (int, float)) else f"    Accuracy: {ensemble.get('accuracy', 'N/A')}")
                    print(f"    Brier Score: {ensemble.get('brier_score', 'N/A'):.4f}" if isinstance(ensemble.get('brier_score'), (int, float)) else f"    Brier Score: {ensemble.get('brier_score', 'N/A')}")
                    print(f"    Log Loss: {ensemble.get('log_loss', 'N/A'):.4f}" if isinstance(ensemble.get('log_loss'), (int, float)) else f"    Log Loss: {ensemble.get('log_loss', 'N/A')}")
                    print(f"    ROI (3%): {ensemble.get('roi_3pct', 'N/A'):.2%}" if isinstance(ensemble.get('roi_3pct'), (int, float)) else f"    ROI (3%): {ensemble.get('roi_3pct', 'N/A')}")
                    
                    print(f"\n  Baseline GBM:")
                    print(f"    Accuracy: {gbm.get('accuracy', 'N/A'):.4f}" if isinstance(gbm.get('accuracy'), (int, float)) else f"    Accuracy: {gbm.get('accuracy', 'N/A')}")
                    print(f"    Brier Score: {gbm.get('brier_score', 'N/A'):.4f}" if isinstance(gbm.get('brier_score'), (int, float)) else f"    Brier Score: {gbm.get('brier_score', 'N/A')}")
                    print(f"    Log Loss: {gbm.get('log_loss', 'N/A'):.4f}" if isinstance(gbm.get('log_loss'), (int, float)) else f"    Log Loss: {gbm.get('log_loss', 'N/A')}")
                    print(f"    ROI (3%): {gbm.get('roi_3pct', 'N/A'):.2%}" if isinstance(gbm.get('roi_3pct'), (int, float)) else f"    ROI (3%): {gbm.get('roi_3pct', 'N/A')}")
                    
                    if 'improvements' in split_data:
                        impr = split_data['improvements']
                        print(f"\n  Improvements (Ensemble - GBM):")
                        print(f"    Accuracy: {impr.get('accuracy', 'N/A'):+.4f}" if isinstance(impr.get('accuracy'), (int, float)) else f"    Accuracy: {impr.get('accuracy', 'N/A')}")
                        print(f"    Brier Score: {impr.get('brier_score', 'N/A'):+.4f}" if isinstance(impr.get('brier_score'), (int, float)) else f"    Brier Score: {impr.get('brier_score', 'N/A')}")
                        print(f"    Log Loss: {impr.get('log_loss', 'N/A'):+.4f}" if isinstance(impr.get('log_loss'), (int, float)) else f"    Log Loss: {impr.get('log_loss', 'N/A')}")
                        print(f"    ROI (3%): {impr.get('roi_3pct', 'N/A'):+.2%}" if isinstance(impr.get('roi_3pct'), (int, float)) else f"    ROI (3%): {impr.get('roi_3pct', 'N/A')}")
    else:
        print("  No comparison data found.")
    
    # Feature Importance
    print(f"\n{'=' * 80}")
    print("FEATURE IMPORTANCE / SHAP VALUES")
    print("=" * 80)
    if audit_results['feature_importance']:
        if 'shap_values_shape' in audit_results['feature_importance']:
            print(f"  SHAP Values Shape: {audit_results['feature_importance']['shap_values_shape']}")
            print(f"  SHAP Values Mean |Value|: {audit_results['feature_importance'].get('shap_values_mean_abs', 'N/A')}")
        
        # Try to extract top features
        if isinstance(audit_results['feature_importance'], dict):
            # Look for feature importance scores
            fi_items = [(k, v) for k, v in audit_results['feature_importance'].items() 
                       if isinstance(v, (int, float)) and k != 'shap_values_mean_abs']
            if fi_items:
                fi_items.sort(key=lambda x: abs(x[1]), reverse=True)
                print(f"\n  Top 5 Features/Models:")
                for i, (name, importance) in enumerate(fi_items[:5], 1):
                    print(f"    {i}. {name}: {importance:.4f}")
    else:
        print("  No feature importance or SHAP values found.")
    
    # Warnings and Errors
    if audit_results['warnings']:
        print(f"\n{'=' * 80}")
        print("WARNINGS")
        print("=" * 80)
        for warning in audit_results['warnings']:
            print(f"  ⚠ {warning}")
    
    if audit_results['errors']:
        print(f"\n{'=' * 80}")
        print("ERRORS")
        print("=" * 80)
        for error in audit_results['errors']:
            print(f"  ❌ {error}")
    
    # Deployment Readiness
    print(f"\n{'=' * 80}")
    print("DEPLOYMENT READINESS")
    print("=" * 80)
    
    required_files = ['model', 'predictions_test']
    has_required = all(
        any(f['type'] == req for f in audit_results['files_found'])
        for req in required_files
    )
    
    if has_required:
        print("  ✓ Core artifacts present (model + test predictions)")
    else:
        print("  ✗ Missing core artifacts required for deployment")
        missing_req = [req for req in required_files 
                      if not any(f['type'] == req for f in audit_results['files_found'])]
        print(f"    Missing: {', '.join(missing_req)}")
    
    print("\n" + "=" * 80)


def main():
    """Main audit function."""
    project_root = Path(__file__).parent.parent
    
    # Check both possible locations
    locations = [
        project_root / "artifacts" / "models" / "nfl_stacked_ensemble",
        project_root / "models" / "artifacts" / "nfl_stacked_ensemble",
    ]
    
    for artifacts_dir in locations:
        if artifacts_dir.exists():
            audit_results = audit_ensemble_results(artifacts_dir)
            print_audit_report(audit_results)
            return
    
    # If neither exists, check what we have
    print("\n" + "=" * 80)
    print("STACKED ENSEMBLE EXPERIMENT AUDIT REPORT")
    print("=" * 80)
    print("\n❌ No ensemble results directory found.")
    print("\nChecked locations:")
    for loc in locations:
        print(f"  - {loc}")
    print("\nTo generate results, run:")
    print("  python3 scripts/train_and_eval_stacked_ensemble.py")
    print("  OR")
    print("  python3 -m models.training.trainer --model stacking_ensemble --config config/models/nfl_stacked_ensemble.yaml --output-dir artifacts/models/nfl_stacked_ensemble")


if __name__ == "__main__":
    main()

