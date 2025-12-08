"""
Model Evaluation Module

Evaluates trained models on holdout datasets.

Usage:
    python -m models.evaluation.evaluate --model ensemble --dataset 2024
    python -m models.evaluation.evaluate --model ensemble --dataset 2024 --model-path artifacts/models/nfl_stacked_ensemble
"""

import sys
import argparse
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.base import BaseModel
from models.training.trainer import load_features, load_backtest_config, split_by_season
from eval.backtest import evaluate_model, compute_market_implied_probabilities
from eval.metrics import accuracy, brier_score, log_loss, calibration_buckets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_model_path(model_type: str, model_path: Optional[Path] = None) -> Path:
    """
    Find the path to a trained model.
    
    Args:
        model_type: Model type (ensemble, ft_transformer, tabnet, etc.)
        model_path: Optional explicit path to model directory or file
    
    Returns:
        Path to model file
    """
    project_root = Path(__file__).parent.parent.parent
    
    # Normalize model type
    if model_type == 'ensemble':
        model_type = 'stacking_ensemble'
    
    # If explicit path provided
    if model_path:
        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = project_root / model_path
        
        # If it's a directory, look for model file
        if model_path.is_dir():
            # Try common model filenames
            for filename in ['ensemble_v1.pkl', 'model.pkl', f'{model_type}.pkl']:
                candidate = model_path / filename
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(
                f"Model file not found in directory {model_path}. "
                f"Expected one of: ensemble_v1.pkl, model.pkl, {model_type}.pkl"
            )
        elif model_path.exists():
            return model_path
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Default: look in artifacts/models/
    default_dirs = {
        'stacking_ensemble': 'nfl_stacked_ensemble',
        'ft_transformer': 'nfl_ft_transformer',
        'tabnet': 'nfl_tabnet',
        'lr': 'nfl_baseline',
        'gbm': 'nfl_baseline',
    }
    
    model_dir_name = default_dirs.get(model_type, f'nfl_{model_type}')
    artifacts_dir = project_root / "artifacts" / "models" / model_dir_name
    
    # Try to find model file
    for filename in ['ensemble_v1.pkl', 'model.pkl', f'{model_type}.pkl']:
        candidate = artifacts_dir / filename
        if candidate.exists():
            return candidate
    
    # For baseline models, try specific names
    if model_type in ['lr', 'logistic']:
        candidate = artifacts_dir / "logit.pkl"
        if candidate.exists():
            return candidate
    elif model_type == 'gbm':
        candidate = artifacts_dir / "gbm.pkl"
        if candidate.exists():
            return candidate
    
    raise FileNotFoundError(
        f"Model not found for type '{model_type}'. "
        f"Looked in: {artifacts_dir}. "
        f"Please train the model first or specify --model-path."
    )


def smart_load_base_model(path: Path) -> BaseModel:
    """
    Smart loader that detects model type and uses appropriate loader.
    
    Args:
        path: Path to model file
    
    Returns:
        Loaded model instance
    """
    import pickle
    
    # Peek at file to detect format
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    # Check for different model formats
    if isinstance(data, BaseModel):
        # Already a model instance
        return data
    elif isinstance(data, dict):
        # Dict format - need to determine type
        if 'config' in data:
            config = data['config']
            # Check for TabNet
            if 'n_steps' in config and 'n_d' in config:
                from models.architectures.tabnet import TabNetModel
                return TabNetModel.load(path)
            # Check for FT-Transformer
            elif 'd_model' in config or 'n_layers' in config:
                from models.architectures.ft_transformer import FTTransformerModel
                return FTTransformerModel.load(path)
            # Check for Gradient Boosting
            elif 'n_estimators' in config or 'max_depth' in config:
                from models.architectures.gradient_boosting import GradientBoostingModel
                return GradientBoostingModel.load(path)
            # Check for Logistic Regression
            elif 'C' in config or 'penalty' in config:
                from models.architectures.logistic_regression import LogisticRegressionModel
                return LogisticRegressionModel.load(path)
        # Check for CalibratedModel
        if 'method' in data and 'base_model_path' in data:
            from models.calibration import CalibratedModel
            return CalibratedModel.load(path)
    
    # Fallback to standard load
    return BaseModel.load(path)


def load_model(model_path: Path) -> BaseModel:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to model file
    
    Returns:
        Loaded model instance
    """
    logger.info(f"Loading model from: {model_path}")
    
    # Try to detect model type and use appropriate loader
    import pickle
    
    # First, peek at what's in the file to determine loader
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        # Check if it's a StackingEnsemble dict format
        if isinstance(data, dict) and 'config' in data and 'base_model_paths' in data:
            from models.architectures.stacking_ensemble import StackingEnsemble
            logger.info("Detected StackingEnsemble format, using custom loader")
            # Use smart loader for base models
            model = StackingEnsemble.load(model_path, base_model_loader=smart_load_base_model)
        # Check if it's a CalibratedModel dict format
        elif isinstance(data, dict) and 'method' in data and 'base_model_path' in data:
            from models.calibration import CalibratedModel
            logger.info("Detected CalibratedModel format, using custom loader")
            model = CalibratedModel.load(model_path, base_model_loader=smart_load_base_model)
        else:
            # Standard BaseModel format (already loaded in data)
            if isinstance(data, BaseModel):
                model = data
            else:
                # Try smart loader
                model = smart_load_base_model(model_path)
    except Exception as e:
        logger.warning(f"Error detecting format, trying smart load: {e}")
        # Fallback to smart loader
        model = smart_load_base_model(model_path)
    
    logger.info(f"✓ Model loaded: {type(model).__name__}")
    return model


def evaluate_on_dataset(
    model: BaseModel,
    dataset_season: int,
    feature_table: Optional[str] = None,
) -> dict:
    """
    Evaluate model on a specific dataset (season).
    
    Args:
        model: Trained model
        dataset_season: Season to evaluate on (e.g., 2024)
        feature_table: Feature table name (if None, uses backtest config)
    
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("=" * 60)
    logger.info(f"Evaluating model on {dataset_season} holdout dataset")
    logger.info("=" * 60)
    
    # Load backtest config for feature table
    backtest_config = load_backtest_config()
    if feature_table is None:
        feature_table = backtest_config.get("feature_table", "baseline")
    
    logger.info(f"Loading features from table: {feature_table}")
    X, y, feature_cols, df = load_features(feature_table=feature_table)
    
    # Filter to dataset season
    dataset_mask = df["season"] == dataset_season
    if dataset_mask.sum() == 0:
        raise ValueError(
            f"No data found for season {dataset_season}. "
            f"Available seasons: {sorted(df['season'].unique())}"
        )
    
    X_dataset = X[dataset_mask].copy()
    y_dataset = y[dataset_mask].copy()
    df_dataset = df[dataset_mask].copy()
    
    logger.info(f"✓ Loaded {len(X_dataset)} games from {dataset_season}")
    
    # Make predictions
    logger.info("Making predictions...")
    y_pred_proba = model.predict_proba(X_dataset)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate basic metrics
    acc = accuracy(y_dataset, y_pred)
    brier = brier_score(y_dataset, y_pred_proba)
    logloss = log_loss(y_dataset, y_pred_proba)
    
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Metrics")
    logger.info("=" * 60)
    logger.info(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    logger.info(f"Brier Score: {brier:.4f}")
    logger.info(f"Log Loss: {logloss:.4f}")
    
    # Calculate calibration
    logger.info("\nCalculating calibration...")
    cal_buckets_df = calibration_buckets(y_dataset, y_pred_proba, n_bins=10)
    logger.info("Calibration buckets:")
    for _, row in cal_buckets_df.iterrows():
        logger.info(
            f"  [{row['bin_min']:.2f}, {row['bin_max']:.2f}]: "
            f"Predicted={row['predicted_freq']:.3f}, "
            f"Actual={row['actual_freq']:.3f}, "
            f"Count={row['count']}, "
            f"Error={row['calibration_error']:.3f}"
        )
    # Convert to list of dicts for results
    cal_buckets = cal_buckets_df.to_dict('records')
    
    # Calculate ROI vs market (if market data available)
    logger.info("\nCalculating ROI vs market...")
    try:
        p_market = compute_market_implied_probabilities(df_dataset)
        edge_thresholds = backtest_config.get("roi", {}).get("edge_thresholds", [0.03, 0.05])
        
        roi_results = {}
        for edge_threshold in edge_thresholds:
            from eval.backtest import calculate_roi
        # Convert to numpy arrays if needed
        y_arr = np.asarray(y_dataset)
        proba_arr = np.asarray(y_pred_proba)
        market_arr = np.asarray(p_market)
        
        roi_dict = calculate_roi(
            y_arr,
            proba_arr,
            market_arr,
            edge_threshold=edge_threshold,
        )
        roi_results[f"edge_{edge_threshold}"] = roi_dict
        logger.info(
            f"\nEdge threshold: {edge_threshold:.2f}"
        )
        logger.info(f"  Total bets: {roi_dict['total_bets']}")
        logger.info(f"  Wins: {roi_dict['wins']}")
        logger.info(f"  Losses: {roi_dict['losses']}")
        logger.info(f"  ROI: {roi_dict['roi']:.2%}")
        logger.info(f"  Profit: ${roi_dict['profit']:.2f}")
    except Exception as e:
        logger.warning(f"Could not calculate ROI vs market: {e}")
        roi_results = {}
    
    # Use evaluate_model for comprehensive results
    logger.info("\nRunning comprehensive evaluation...")
    edge_thresholds = backtest_config.get("roi", {}).get("edge_thresholds", [0.03, 0.05])
    comprehensive_results = evaluate_model(
        model,
        X_dataset,
        y_dataset,
        df_dataset,
        f"{dataset_season}_holdout",
        edge_thresholds,
    )
    
    # Combine results
    results = {
        "dataset_season": dataset_season,
        "n_games": len(X_dataset),
        "accuracy": acc,
        "brier_score": brier,
        "log_loss": logloss,
        "calibration_buckets": cal_buckets,
        "roi_results": roi_results,
        "comprehensive_results": comprehensive_results,
    }
    
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Complete!")
    logger.info("=" * 60)
    
    return results


def main():
    """CLI entrypoint for model evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained models on holdout datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate ensemble on 2024 holdout
  python -m models.evaluation.evaluate --model ensemble --dataset 2024
  
  # Evaluate with custom model path
  python -m models.evaluation.evaluate --model ensemble --dataset 2024 --model-path artifacts/models/nfl_stacked_ensemble
  
  # Evaluate FT-Transformer on 2023
  python -m models.evaluation.evaluate --model ft_transformer --dataset 2023
        """
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model type (ensemble, ft_transformer, tabnet, lr, gbm)'
    )
    parser.add_argument(
        '--dataset',
        type=int,
        required=True,
        help='Dataset season to evaluate on (e.g., 2024)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to model file or directory (if not provided, uses default artifacts location)'
    )
    parser.add_argument(
        '--feature-table',
        type=str,
        default=None,
        help='Feature table name (if not provided, uses backtest config)'
    )
    
    args = parser.parse_args()
    
    # Find and load model
    model_path = find_model_path(args.model, Path(args.model_path) if args.model_path else None)
    model = load_model(model_path)
    
    # Evaluate on dataset
    results = evaluate_on_dataset(
        model,
        args.dataset,
        feature_table=args.feature_table,
    )
    
    logger.info(f"\nEvaluation complete. Results for {args.dataset}:")
    logger.info(f"  Accuracy: {results['accuracy']:.4f}")
    logger.info(f"  Brier Score: {results['brier_score']:.4f}")
    logger.info(f"  Log Loss: {results['log_loss']:.4f}")


if __name__ == "__main__":
    main()

