"""
Train and evaluate stacked ensemble v2 (FT-Transformer + GBM only), 
comparing to v1 (FT-Transformer + TabNet + GBM).
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
from models.architectures.stacking_ensemble import StackingEnsemble
from models.architectures.ft_transformer import FTTransformerModel
from models.architectures.tabnet import TabNetModel
from models.base import BaseModel
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
    """Save model predictions to parquet file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get predictions
    p_pred = model.predict_proba(X)
    y_pred = (p_pred >= 0.5).astype(int)
    
    # Align indices
    if not X.index.equals(df.index):
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
    ensemble_v2_model,
    ensemble_v1_model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    df_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_test: pd.DataFrame,
    output_dir: Path,
):
    """Compare ensemble v2 (FT-Transformer + GBM) vs v1 (FT-Transformer + TabNet + GBM)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "=" * 60)
    logger.info("Model Comparison: Ensemble V2 vs V1")
    logger.info("=" * 60)
    
    # Evaluate both models
    edge_thresholds = [0.03, 0.05]
    
    logger.info("\nEvaluating Ensemble V2 (FT-Transformer + GBM)...")
    v2_val_results = evaluate_model(
        ensemble_v2_model, X_val, y_val, df_val, "validation", edge_thresholds
    )
    v2_test_results = evaluate_model(
        ensemble_v2_model, X_test, y_test, df_test, "test", edge_thresholds
    )
    
    logger.info("\nEvaluating Ensemble V1 (FT-Transformer + TabNet + GBM)...")
    v1_val_results = evaluate_model(
        ensemble_v1_model, X_val, y_val, df_val, "validation", edge_thresholds
    )
    v1_test_results = evaluate_model(
        ensemble_v1_model, X_test, y_test, df_test, "test", edge_thresholds
    )
    
    # Create comparison summary
    comparison = {
        'validation': {
            'ensemble_v2': {
                'accuracy': v2_val_results['accuracy'],
                'brier_score': v2_val_results['brier_score'],
                'log_loss': v2_val_results['log_loss'],
                'mean_calibration_error': v2_val_results['mean_calibration_error'],
                'roi_3pct': v2_val_results['roi_results'].get('roi_threshold_0.03', {}).get('roi', 0.0),
                'roi_5pct': v2_val_results['roi_results'].get('roi_threshold_0.05', {}).get('roi', 0.0),
            },
            'ensemble_v1': {
                'accuracy': v1_val_results['accuracy'],
                'brier_score': v1_val_results['brier_score'],
                'log_loss': v1_val_results['log_loss'],
                'mean_calibration_error': v1_val_results['mean_calibration_error'],
                'roi_3pct': v1_val_results['roi_results'].get('roi_threshold_0.03', {}).get('roi', 0.0),
                'roi_5pct': v1_val_results['roi_results'].get('roi_threshold_0.05', {}).get('roi', 0.0),
            },
        },
        'test': {
            'ensemble_v2': {
                'accuracy': v2_test_results['accuracy'],
                'brier_score': v2_test_results['brier_score'],
                'log_loss': v2_test_results['log_loss'],
                'mean_calibration_error': v2_test_results['mean_calibration_error'],
                'roi_3pct': v2_test_results['roi_results'].get('roi_threshold_0.03', {}).get('roi', 0.0),
                'roi_5pct': v2_test_results['roi_results'].get('roi_threshold_0.05', {}).get('roi', 0.0),
            },
            'ensemble_v1': {
                'accuracy': v1_test_results['accuracy'],
                'brier_score': v1_test_results['brier_score'],
                'log_loss': v1_test_results['log_loss'],
                'mean_calibration_error': v1_test_results['mean_calibration_error'],
                'roi_3pct': v1_test_results['roi_results'].get('roi_threshold_0.03', {}).get('roi', 0.0),
                'roi_5pct': v1_test_results['roi_results'].get('roi_threshold_0.05', {}).get('roi', 0.0),
            },
        },
    }
    
    # Calculate differences (V2 - V1)
    for split in ['validation', 'test']:
        v2 = comparison[split]['ensemble_v2']
        v1 = comparison[split]['ensemble_v1']
        
        comparison[split]['differences'] = {
            'accuracy': v2['accuracy'] - v1['accuracy'],
            'brier_score': v2['brier_score'] - v1['brier_score'],  # Lower is better
            'log_loss': v2['log_loss'] - v1['log_loss'],  # Lower is better
            'mean_calibration_error': v2['mean_calibration_error'] - v1['mean_calibration_error'],  # Lower is better
            'roi_3pct': v2['roi_3pct'] - v1['roi_3pct'],
            'roi_5pct': v2['roi_5pct'] - v1['roi_5pct'],
        }
    
    # Save comparison
    comparison_path = output_dir / "comparison_vs_v1.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"\nSaved comparison results to {comparison_path}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Comparison Summary: V2 (FT-Transformer + GBM) vs V1 (FT-Transformer + TabNet + GBM)")
    logger.info("=" * 60)
    
    logger.info("\nValidation Set:")
    logger.info(f"  Accuracy: V2={comparison['validation']['ensemble_v2']['accuracy']:.4f}, "
                f"V1={comparison['validation']['ensemble_v1']['accuracy']:.4f}, "
                f"Δ={comparison['validation']['differences']['accuracy']:+.4f}")
    logger.info(f"  Brier: V2={comparison['validation']['ensemble_v2']['brier_score']:.4f}, "
                f"V1={comparison['validation']['ensemble_v1']['brier_score']:.4f}, "
                f"Δ={comparison['validation']['differences']['brier_score']:+.4f}")
    logger.info(f"  Log Loss: V2={comparison['validation']['ensemble_v2']['log_loss']:.4f}, "
                f"V1={comparison['validation']['ensemble_v1']['log_loss']:.4f}, "
                f"Δ={comparison['validation']['differences']['log_loss']:+.4f}")
    logger.info(f"  ROI (3%): V2={comparison['validation']['ensemble_v2']['roi_3pct']:.2%}, "
                f"V1={comparison['validation']['ensemble_v1']['roi_3pct']:.2%}, "
                f"Δ={comparison['validation']['differences']['roi_3pct']:+.2%}")
    
    logger.info("\nTest Set:")
    logger.info(f"  Accuracy: V2={comparison['test']['ensemble_v2']['accuracy']:.4f}, "
                f"V1={comparison['test']['ensemble_v1']['accuracy']:.4f}, "
                f"Δ={comparison['test']['differences']['accuracy']:+.4f}")
    logger.info(f"  Brier: V2={comparison['test']['ensemble_v2']['brier_score']:.4f}, "
                f"V1={comparison['test']['ensemble_v1']['brier_score']:.4f}, "
                f"Δ={comparison['test']['differences']['brier_score']:+.4f}")
    logger.info(f"  Log Loss: V2={comparison['test']['ensemble_v2']['log_loss']:.4f}, "
                f"V1={comparison['test']['ensemble_v1']['log_loss']:.4f}, "
                f"Δ={comparison['test']['differences']['log_loss']:+.4f}")
    logger.info(f"  ROI (3%): V2={comparison['test']['ensemble_v2']['roi_3pct']:.2%}, "
                f"V1={comparison['test']['ensemble_v1']['roi_3pct']:.2%}, "
                f"Δ={comparison['test']['differences']['roi_3pct']:+.2%}")
    
    return comparison


def extract_meta_model_importance(ensemble_path: Path, output_path: Path):
    """Extract logistic regression coefficients from meta-model."""
    ensemble = StackingEnsemble.load(ensemble_path)
    
    importance = {}
    base_model_names = list(ensemble.base_models.keys())
    
    if ensemble.meta_model_type == 'logistic' and ensemble.meta_model is not None:
        if hasattr(ensemble.meta_model, 'coef_'):
            coef = ensemble.meta_model.coef_[0]
            for i, model_name in enumerate(base_model_names):
                if i < len(coef):
                    importance[model_name] = float(coef[i])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(importance, f, indent=2)
    
    logger.info(f"Saved feature importance to {output_path}")
    return importance


def main():
    """Main training and evaluation pipeline."""
    logger.info("=" * 60)
    logger.info("Stacked Ensemble V2 Training and Evaluation")
    logger.info("=" * 60)
    
    # Set up paths
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "models" / "nfl_stacked_ensemble_v2.yaml"
    artifacts_dir = project_root / "artifacts" / "models" / "nfl_stacked_ensemble_v2"
    v1_ensemble_path = project_root / "artifacts" / "models" / "nfl_stacked_ensemble" / "ensemble_v1.pkl"
    
    # Step 1: Train ensemble v2
    logger.info("\n[Step 1/5] Training Stacked Ensemble V2 (FT-Transformer + GBM)...")
    ensemble_v2_model, X_train, y_train, X_val, y_val, X_test, y_test = run_advanced_training_pipeline(
        model_type='stacking_ensemble',
        config_path=config_path,
        artifacts_dir=artifacts_dir,
        apply_calibration_flag=True,
    )
    logger.info("✓ Ensemble V2 training complete")
    
    # Step 2: Load ensemble v1
    logger.info("\n[Step 2/5] Loading Ensemble V1...")
    if not v1_ensemble_path.exists():
        raise FileNotFoundError(
            f"Ensemble V1 not found at {v1_ensemble_path}. "
            "Please train V1 first."
        )
    
    # Custom loader to handle different model types
    def custom_base_model_loader(path):
        """Load base model with proper type detection."""
        path = Path(path)
        # Try to detect model type from filename or load and check
        if 'ft_transformer' in str(path):
            return FTTransformerModel.load(path)
        elif 'tabnet' in str(path):
            # TabNet may have path issues - try to fix
            model = TabNetModel.load(path)
            # Force CPU if CUDA not compatible
            if hasattr(model, 'device'):
                if model.device == 'cuda':
                    # Check if CUDA actually works
                    import torch
                    try:
                        test_tensor = torch.zeros(1).cuda()
                        _ = test_tensor + 1
                        del test_tensor
                        torch.cuda.empty_cache()
                    except Exception:
                        # CUDA not usable, force CPU
                        model.device = 'cpu'
                        if hasattr(model, 'model') and model.model is not None:
                            if hasattr(model.model, 'device'):
                                model.model.device = 'cpu'
                            # Move model to CPU if it's a torch model
                            if hasattr(model.model, 'to'):
                                model.model = model.model.to('cpu')
            return model
        elif 'gbm' in str(path):
            return GradientBoostingModel.load(path)
        else:
            # Fallback to generic load
            return BaseModel.load(path)
    
    ensemble_v1_model = StackingEnsemble.load(v1_ensemble_path, base_model_loader=custom_base_model_loader)
    logger.info("✓ Ensemble V1 loaded")
    
    # Step 3: Load full dataframe for predictions
    logger.info("\n[Step 3/5] Loading full dataframe...")
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
    logger.info("\n[Step 4/5] Saving predictions...")
    save_predictions(ensemble_v2_model, X_train, y_train, df_train, "train", artifacts_dir)
    save_predictions(ensemble_v2_model, X_val, y_val, df_val, "val", artifacts_dir)
    save_predictions(ensemble_v2_model, X_test, y_test, df_test, "test", artifacts_dir)
    logger.info("✓ Predictions saved")
    
    # Step 5: Compare to v1
    logger.info("\n[Step 5/5] Comparing V2 vs V1...")
    comparison = compare_models(
        ensemble_v2_model,
        ensemble_v1_model,
        X_val, y_val, df_val,
        X_test, y_test, df_test,
        artifacts_dir,
    )
    logger.info("✓ Comparison complete")
    
    # Step 6: Extract feature importance
    logger.info("\n[Step 6/6] Extracting feature importance...")
    importance_path = artifacts_dir / "feature_importance.json"
    importance = extract_meta_model_importance(artifacts_dir / "ensemble_v1.pkl", importance_path)
    logger.info("✓ Feature importance extracted")
    
    logger.info("\n" + "=" * 60)
    logger.info("Training and Evaluation Complete!")
    logger.info("=" * 60)
    logger.info(f"\nResults saved to: {artifacts_dir}")
    logger.info(f"  - Model: ensemble_v1.pkl")
    logger.info(f"  - Predictions: predictions_train/val/test.parquet")
    logger.info(f"  - Comparison: comparison_vs_v1.json")
    logger.info(f"  - Feature Importance: feature_importance.json")


if __name__ == "__main__":
    main()

