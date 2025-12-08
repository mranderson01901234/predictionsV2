#!/usr/bin/env python3
"""
Production Training Script for NFL Prediction Model

This script implements best practices for production-ready model training:
- Walk-forward validation (multiple test sets)
- Hyperparameter tuning (Optuna)
- Feature selection (importance-based)
- Optimal calibration (method comparison)
- Comprehensive evaluation and reporting

Usage:
    python scripts/train_production.py --model gbm --feature-table baseline
    python scripts/train_production.py --model stacking_ensemble --tune-hyperparameters
    python scripts/train_production.py --model gbm --walk-forward --output-dir artifacts/production_v1
"""

import argparse
import logging
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pandas as pd
import numpy as np
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.training.trainer import (
    load_features,
    split_by_season,
    train_model,
    apply_calibration,
    load_config,
    load_backtest_config,
)
from models.training.walk_forward import WalkForwardValidator, WalkForwardResult
from models.calibration import CalibratedModel, compute_calibration_metrics
from models.base import BaseModel
from eval.metrics import accuracy, brier_score, log_loss, calibration_buckets
from features.feature_table_registry import get_feature_table_path, validate_feature_table_exists

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """
    Hyperparameter tuning using Optuna for Bayesian optimization.
    """
    
    def __init__(
        self,
        model_type: str,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        random_state: int = 42,
    ):
        self.model_type = model_type
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_state = random_state
        
    def tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters for the specified model type.
        
        Returns:
            Dictionary with best hyperparameters
        """
        try:
            import optuna
        except ImportError:
            logger.error("Optuna not installed. Install with: pip install optuna")
            return {}
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Hyperparameter Tuning: {self.model_type}")
        logger.info(f"Trials: {self.n_trials}, Timeout: {self.timeout}s")
        logger.info(f"{'='*60}")
        
        def objective(trial):
            """Objective function for Optuna."""
            if self.model_type in ['gbm', 'gradient_boosting', 'xgboost']:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                    'random_state': self.random_state,
                }
                
                from models.architectures.gradient_boosting import GradientBoostingModel
                model = GradientBoostingModel(**params)
                model.fit(X_train, y_train)
                
            elif self.model_type in ['lr', 'logistic', 'logistic_regression']:
                params = {
                    'C': trial.suggest_float('C', 0.01, 100.0, log=True),
                    'max_iter': trial.suggest_int('max_iter', 100, 5000),
                    'random_state': self.random_state,
                }
                
                from models.architectures.logistic_regression import LogisticRegressionModel
                model = LogisticRegressionModel(**params)
                model.fit(X_train, y_train)
                
            else:
                logger.warning(f"Hyperparameter tuning not implemented for {self.model_type}")
                return 0.0
            
            # Evaluate on validation set
            y_pred_proba = model.predict_proba(X_val)
            score = -brier_score(y_val.values, y_pred_proba)  # Negative because Optuna minimizes
            
            return score
        
        study = optuna.create_study(
            direction='maximize',
            study_name=f'{self.model_type}_tuning',
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
        )
        
        best_params = study.best_params
        best_score = study.best_value
        
        logger.info(f"\nBest hyperparameters:")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value}")
        logger.info(f"Best Brier Score: {-best_score:.4f}")
        
        return best_params


class FeatureSelector:
    """
    Feature selection based on importance scores.
    """
    
    def __init__(
        self,
        method: str = 'importance',
        top_k: Optional[int] = None,
        importance_threshold: Optional[float] = None,
    ):
        self.method = method
        self.top_k = top_k
        self.importance_threshold = importance_threshold
        
    def select_features(
        self,
        model: BaseModel,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> List[str]:
        """
        Select features based on model importance.
        
        Returns:
            List of selected feature names
        """
        logger.info(f"\n{'='*60}")
        logger.info("Feature Selection")
        logger.info(f"{'='*60}")
        
        # Get feature importance
        if hasattr(model, 'get_feature_importance'):
            importances = model.get_feature_importance()
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'feature_importances_'):
            importances = model.base_model.feature_importances_
        else:
            logger.warning("Model does not support feature importance. Using all features.")
            return list(X_train.columns)
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importances,
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\nTop 20 Features by Importance:")
        for idx, row in importance_df.head(20).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Select features
        if self.top_k:
            selected = importance_df.head(self.top_k)['feature'].tolist()
            logger.info(f"\nSelected top {self.top_k} features")
        elif self.importance_threshold:
            selected = importance_df[importance_df['importance'] >= self.importance_threshold]['feature'].tolist()
            logger.info(f"\nSelected {len(selected)} features with importance >= {self.importance_threshold}")
        else:
            # Use all features
            selected = list(X_train.columns)
            logger.info(f"\nUsing all {len(selected)} features")
        
        return selected


class CalibrationOptimizer:
    """
    Optimize calibration method and parameters.
    """
    
    def __init__(self, methods: List[str] = ['platt', 'isotonic', 'temperature']):
        self.methods = methods
        
    def optimize(
        self,
        model: BaseModel,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
    ) -> Tuple[str, CalibratedModel]:
        """
        Find best calibration method.
        
        Returns:
            Tuple of (best_method, calibrated_model)
        """
        logger.info(f"\n{'='*60}")
        logger.info("Calibration Optimization")
        logger.info(f"{'='*60}")
        
        best_method = None
        best_brier = float('inf')
        best_calibrated = None
        
        for method in self.methods:
            logger.info(f"\nTesting {method} calibration...")
            calibrated = CalibratedModel(base_model=model, method=method)
            calibrated.fit_calibration(X_cal, y_cal)
            
            y_pred_proba = calibrated.predict_proba(X_cal)
            brier = brier_score(y_cal.values, y_pred_proba)
            
            logger.info(f"  {method}: Brier = {brier:.4f}")
            
            if brier < best_brier:
                best_brier = brier
                best_method = method
                best_calibrated = calibrated
        
        logger.info(f"\nBest calibration method: {best_method} (Brier: {best_brier:.4f})")
        
        return best_method, best_calibrated


def train_with_walk_forward(
    model_type: str,
    feature_table: str,
    config_path: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
    tune_hyperparameters: bool = False,
    apply_feature_selection: bool = False,
    optimize_calibration: bool = True,
    min_train_seasons: int = 4,
    start_test_season: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[WalkForwardResult]]:
    """
    Train model with walk-forward validation.
    
    Returns:
        Tuple of (summary_df, results_list)
    """
    logger.info("="*60)
    logger.info("PRODUCTION TRAINING WITH WALK-FORWARD VALIDATION")
    logger.info("="*60)
    
    # Load data
    logger.info(f"\nLoading features from table: {feature_table}")
    validate_feature_table_exists(feature_table)
    X, y, feature_cols, df = load_features(feature_table=feature_table)
    
    logger.info(f"Loaded {len(df)} games")
    logger.info(f"Seasons: {sorted(df['season'].unique())}")
    logger.info(f"Features: {len(feature_cols)}")
    
    # Load config
    if config_path:
        config = load_config(config_path)
    else:
        config = {}
    
    # Set up artifacts directory
    if artifacts_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifacts_dir = project_root / "models" / "artifacts" / f"production_{model_type}_{timestamp}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Artifacts directory: {artifacts_dir}")
    
    # Hyperparameter tuning (if enabled)
    best_params = {}
    if tune_hyperparameters:
        # Use first few seasons for tuning
        backtest_config = load_backtest_config()
        train_seasons = backtest_config['splits']['train_seasons']
        val_season = backtest_config['splits']['validation_season']
        
        X_train_tune, y_train_tune, X_val_tune, y_val_tune, _, _ = split_by_season(
            X, y, df, train_seasons, val_season, val_season + 1
        )
        
        tuner = HyperparameterTuner(model_type, n_trials=50)
        best_params = tuner.tune(X_train_tune, y_train_tune, X_val_tune, y_val_tune)
        
        # Update config with best params
        if model_type in ['gbm', 'gradient_boosting']:
            config.setdefault('models', {}).setdefault('gradient_boosting', {}).update(best_params)
        elif model_type in ['lr', 'logistic']:
            config.setdefault('models', {}).setdefault('logistic_regression', {}).update(best_params)
    
    # Create model factory
    def model_factory():
        if model_type in ['gbm', 'gradient_boosting']:
            from models.architectures.gradient_boosting import GradientBoostingModel
            gbm_config = config.get('models', {}).get('gradient_boosting', {})
            if best_params:
                gbm_config.update(best_params)
            return GradientBoostingModel(**gbm_config)
        elif model_type in ['lr', 'logistic']:
            from models.architectures.logistic_regression import LogisticRegressionModel
            lr_config = config.get('models', {}).get('logistic_regression', {})
            if best_params:
                lr_config.update(best_params)
            return LogisticRegressionModel(**lr_config)
        elif model_type == 'stacking_ensemble':
            from models.training.trainer import train_stacking_ensemble
            # Will be handled differently
            return None
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    # Run walk-forward validation
    validator = WalkForwardValidator(
        features_df=df,
        model_factory=model_factory,
        min_train_seasons=min_train_seasons,
        calibration_method='isotonic' if optimize_calibration else None,
        feature_cols=feature_cols,
    )
    
    results = validator.run_all_splits(
        start_test_season=start_test_season,
        apply_calibration=optimize_calibration,
    )
    
    summary = validator.summarize_results()
    
    # Save results
    summary.to_csv(artifacts_dir / "walk_forward_summary.csv", index=False)
    
    all_predictions = pd.concat([r.predictions for r in results])
    all_predictions.to_parquet(artifacts_dir / "walk_forward_predictions.parquet")
    
    # Save config
    config_dict = {
        'model_type': model_type,
        'feature_table': feature_table,
        'tune_hyperparameters': tune_hyperparameters,
        'best_params': best_params,
        'n_splits': len(results),
        'mean_accuracy': summary['accuracy'].mean(),
        'mean_brier': summary['brier'].mean(),
        'timestamp': datetime.now().isoformat(),
    }
    with open(artifacts_dir / "training_config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Results saved to: {artifacts_dir}")
    logger.info(f"Mean Accuracy: {summary['accuracy'].mean():.3f} ± {summary['accuracy'].std():.3f}")
    logger.info(f"Mean Brier Score: {summary['brier'].mean():.4f}")
    
    return summary, results


def train_single_split(
    model_type: str,
    feature_table: str,
    config_path: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
    tune_hyperparameters: bool = False,
    apply_feature_selection: bool = False,
    optimize_calibration: bool = True,
) -> BaseModel:
    """
    Train model with single train/val/test split (faster, less robust).
    
    Returns:
        Trained model
    """
    logger.info("="*60)
    logger.info("PRODUCTION TRAINING (SINGLE SPLIT)")
    logger.info("="*60)
    
    # Load data
    logger.info(f"\nLoading features from table: {feature_table}")
    validate_feature_table_exists(feature_table)
    X, y, feature_cols, df = load_features(feature_table=feature_table)
    
    # Load config
    if config_path:
        config = load_config(config_path)
    else:
        config = load_config()  # Default config
    
    # Split data
    backtest_config = load_backtest_config()
    train_seasons = backtest_config['splits']['train_seasons']
    val_season = backtest_config['splits']['validation_season']
    test_season = backtest_config['splits']['test_season']
    
    X_train, y_train, X_val, y_val, X_test, y_test = split_by_season(
        X, y, df, train_seasons, val_season, test_season
    )
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Hyperparameter tuning
    best_params = {}
    if tune_hyperparameters:
        tuner = HyperparameterTuner(model_type, n_trials=50)
        best_params = tuner.tune(X_train, y_train, X_val, y_val)
        
        # Update config
        if model_type in ['gbm', 'gradient_boosting']:
            config.setdefault('models', {}).setdefault('gradient_boosting', {}).update(best_params)
        elif model_type in ['lr', 'logistic']:
            config.setdefault('models', {}).setdefault('logistic_regression', {}).update(best_params)
    
    # Feature selection
    if apply_feature_selection:
        # Train a quick model to get feature importance
        from models.architectures.gradient_boosting import GradientBoostingModel
        temp_model = GradientBoostingModel(n_estimators=50, random_state=42)
        temp_model.fit(X_train, y_train)
        
        selector = FeatureSelector(top_k=50)  # Top 50 features
        selected_features = selector.select_features(temp_model, X_train, y_train)
        
        X_train = X_train[selected_features]
        X_val = X_val[selected_features]
        X_test = X_test[selected_features]
        
        logger.info(f"Selected {len(selected_features)} features")
    
    # Train model
    model = train_model(
        model_type=model_type,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        config=config,
        artifacts_dir=artifacts_dir,
    )
    
    # Calibration optimization
    if optimize_calibration:
        optimizer = CalibrationOptimizer()
        best_method, calibrated_model = optimizer.optimize(model, X_val, y_val)
        model = calibrated_model
    
    # Evaluate on test set
    logger.info(f"\n{'='*60}")
    logger.info("TEST SET EVALUATION")
    logger.info(f"{'='*60}")
    
    y_pred_proba = model.predict_proba(X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    test_accuracy = accuracy(y_test.values, y_pred)
    test_brier = brier_score(y_test.values, y_pred_proba)
    test_log_loss = log_loss(y_test.values, y_pred_proba)
    
    logger.info(f"Accuracy: {test_accuracy:.4f}")
    logger.info(f"Brier Score: {test_brier:.4f}")
    logger.info(f"Log Loss: {test_log_loss:.4f}")
    
    # Calibration metrics
    cal_metrics = compute_calibration_metrics(y_test.values, y_pred_proba)
    logger.info(f"Calibration Error (ECE): {cal_metrics['ece']:.4f}")
    logger.info(f"Max Calibration Error (MCE): {cal_metrics['mce']:.4f}")
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Production training script for NFL prediction model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train GBM with walk-forward validation
  python scripts/train_production.py --model gbm --walk-forward
  
  # Train with hyperparameter tuning
  python scripts/train_production.py --model gbm --tune-hyperparameters
  
  # Train stacking ensemble
  python scripts/train_production.py --model stacking_ensemble --feature-table phase2b
  
  # Single split (faster)
  python scripts/train_production.py --model gbm --no-walk-forward
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['gbm', 'lr', 'stacking_ensemble', 'ft_transformer', 'tabnet'],
        help='Model type to train'
    )
    parser.add_argument(
        '--feature-table',
        type=str,
        default='baseline',
        help='Feature table name (baseline, phase2, phase2b)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to model config file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for artifacts'
    )
    parser.add_argument(
        '--walk-forward',
        action='store_true',
        help='Use walk-forward validation (recommended)'
    )
    parser.add_argument(
        '--no-walk-forward',
        action='store_true',
        help='Use single train/test split (faster)'
    )
    parser.add_argument(
        '--tune-hyperparameters',
        action='store_true',
        help='Tune hyperparameters with Optuna'
    )
    parser.add_argument(
        '--feature-selection',
        action='store_true',
        help='Apply feature selection (top N features)'
    )
    parser.add_argument(
        '--no-calibration',
        action='store_true',
        help='Skip calibration optimization'
    )
    parser.add_argument(
        '--min-train-seasons',
        type=int,
        default=4,
        help='Minimum training seasons for walk-forward'
    )
    parser.add_argument(
        '--start-test-season',
        type=int,
        default=None,
        help='First test season for walk-forward'
    )
    
    args = parser.parse_args()
    
    # Determine validation mode
    use_walk_forward = args.walk_forward or (not args.no_walk_forward)
    
    artifacts_dir = Path(args.output_dir) if args.output_dir else None
    config_path = Path(args.config) if args.config else None
    
    if use_walk_forward:
        summary, results = train_with_walk_forward(
            model_type=args.model,
            feature_table=args.feature_table,
            config_path=config_path,
            artifacts_dir=artifacts_dir,
            tune_hyperparameters=args.tune_hyperparameters,
            apply_feature_selection=args.feature_selection,
            optimize_calibration=not args.no_calibration,
            min_train_seasons=args.min_train_seasons,
            start_test_season=args.start_test_season,
        )
    else:
        model = train_single_split(
            model_type=args.model,
            feature_table=args.feature_table,
            config_path=config_path,
            artifacts_dir=artifacts_dir,
            tune_hyperparameters=args.tune_hyperparameters,
            apply_feature_selection=args.feature_selection,
            optimize_calibration=not args.no_calibration,
        )


if __name__ == "__main__":
    main()

