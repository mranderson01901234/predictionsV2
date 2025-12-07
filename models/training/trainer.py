"""
Unified Model Trainer

Handles data loading, splitting, and training of all model types:
- Baseline models: Logistic Regression, Gradient Boosting, Simple Ensemble
- Advanced models: FT-Transformer, TabNet, Stacking Ensemble

All models inherit from BaseModel and use unified config format.
"""

import pandas as pd
import numpy as np
import yaml
import json
import argparse
from pathlib import Path
from typing import Tuple, List, Optional, Union, Dict, Any
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.architectures.logistic_regression import LogisticRegressionModel
from models.architectures.gradient_boosting import GradientBoostingModel
from models.architectures.ensemble import EnsembleModel
from models.architectures.ft_transformer import FTTransformerModel
from models.architectures.tabnet import TabNetModel
from models.architectures.stacking_ensemble import StackingEnsemble
from models.calibration import CalibratedModel
from models.base import BaseModel
from features.feature_table_registry import get_feature_table_path, validate_feature_table_exists

# Try to import feature registry (optional, for advanced feature selection)
try:
    from features.registry import FeatureRegistry, FeatureGroup
    FEATURE_REGISTRY_AVAILABLE = True
except ImportError:
    FEATURE_REGISTRY_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: Optional[Path] = None) -> dict:
    """
    Load model configuration.
    
    Args:
        config_path: Path to config file. If None, uses default baseline config.
                    Can be absolute or relative to project root.
    
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "models" / "nfl_baseline.yaml"
    else:
        config_path = Path(config_path)
        # If relative path, resolve relative to project root
        if not config_path.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / config_path
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Ensure numeric values are properly typed (YAML sometimes loads 1e-4 as string)
    def convert_numeric(obj):
        if isinstance(obj, dict):
            return {k: convert_numeric(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numeric(item) for item in obj]
        elif isinstance(obj, str):
            # Try to convert scientific notation strings to float
            try:
                if 'e' in obj.lower() or 'E' in obj:
                    return float(obj)
            except (ValueError, AttributeError):
                pass
        return obj
    
    return convert_numeric(config)


def load_backtest_config() -> dict:
    """Load backtest configuration."""
    config_path = Path(__file__).parent.parent.parent / "config" / "evaluation" / "backtest_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_features(
    features_path: Optional[Path] = None,
    feature_table: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.DataFrame]:
    """
    Load game features and create target variable.
    
    Args:
        features_path: Path to feature parquet file (if None, uses config)
        feature_table: Feature table name ("baseline", "phase2", "phase2b")
    
    Returns:
        Tuple of (features_df, target_series, feature_columns, full_dataframe)
        The full_dataframe includes all columns for downstream use (e.g., market data).
    """
    if features_path is None:
        # Get feature table from config if not specified
        if feature_table is None:
            config = load_backtest_config()
            feature_table = config.get("feature_table", "baseline")
        
        # Validate feature table exists
        validate_feature_table_exists(feature_table)
        
        # Get path from registry
        features_path = get_feature_table_path(feature_table)
    
    logger.info(f"Loading features from {features_path} (single load)")
    df = pd.read_parquet(features_path)
    
    # Create target: home_win (1 if home_score > away_score, else 0)
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    # Handle ties as 0 (away win equivalent for binary classification)
    
    # Identify feature columns (exclude leakage and metadata)
    exclude_cols = [
        "game_id",
        "season",
        "week",
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "home_win",  # Target
        "close_spread",  # Keep for ROI calculation but not as feature
        "close_total",  # Keep for ROI calculation but not as feature
        "open_spread",  # Optional, exclude from features
        "open_total",  # Optional, exclude from features
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Ensure we have features
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found!")
    
    logger.info(f"Found {len(feature_cols)} feature columns")
    logger.info(f"Feature columns: {feature_cols[:10]}..." if len(feature_cols) > 10 else f"Feature columns: {feature_cols}")
    
    # Extract features and target
    X = df[feature_cols].copy()
    y = df["home_win"].copy()
    
    # Check for missing values
    missing = X.isna().sum().sum()
    if missing > 0:
        logger.warning(f"Found {missing} missing values in features. Filling with 0.")
        X = X.fillna(0)
    
    logger.info(f"Loaded {len(X)} games")
    logger.info(f"Home win rate: {y.mean():.3f}")
    
    return X, y, feature_cols, df


def get_feature_columns(df: pd.DataFrame, config: dict) -> list:
    """
    Get feature columns based on config.
    
    Supports feature selection via feature_groups if FeatureRegistry is available.
    """
    all_columns = list(df.columns)
    
    # Get groups from config
    groups = config.get('features', {}).get('feature_groups')
    
    if groups and FEATURE_REGISTRY_AVAILABLE:
        group_enums = [FeatureGroup(g) for g in groups]
        allowed = set(FeatureRegistry.get_feature_groups(group_enums))
        columns = [c for c in all_columns if c in allowed]
    else:
        # Use default exclusions (same as load_features)
        exclude_cols = [
            "game_id", "season", "week", "date",
            "home_team", "away_team", "home_score", "away_score",
            "home_win", "close_spread", "close_total",
            "open_spread", "open_total",
        ]
        columns = [c for c in all_columns if c not in exclude_cols]
    
    return columns


def split_by_season(
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    train_seasons: List[int],
    validation_season: int,
    test_season: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split data by season.
    
    Args:
        X: Feature matrix
        y: Target vector
        df: Full dataframe (for season column)
        train_seasons: List of training seasons
        validation_season: Validation season
        test_season: Test season
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    logger.info(f"Splitting data: Train={train_seasons}, Val={validation_season}, Test={test_season}")
    
    # Ensure indices align
    if not X.index.equals(df.index):
        # Reset indices to align
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        df = df.reset_index(drop=True)
    
    train_mask = df["season"].isin(train_seasons)
    val_mask = df["season"] == validation_season
    test_mask = df["season"] == test_season
    
    X_train = X[train_mask].copy()
    y_train = y[train_mask].copy()
    
    X_val = X[val_mask].copy()
    y_val = y[val_mask].copy()
    
    X_test = X[test_mask].copy()
    y_test = y[test_mask].copy()
    
    logger.info(f"Train: {len(X_train)} games")
    logger.info(f"Validation: {len(X_val)} games")
    logger.info(f"Test: {len(X_test)} games")
    
    # Validate no overlap
    train_seasons_set = set(train_seasons)
    assert validation_season not in train_seasons_set, "Validation season in train!"
    assert test_season not in train_seasons_set, "Test season in train!"
    assert validation_season != test_season, "Validation and test seasons overlap!"
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
    artifacts_dir: Path,
) -> LogisticRegressionModel:
    """Train logistic regression model."""
    logger.info("\n" + "=" * 60)
    logger.info("Training Logistic Regression")
    logger.info("=" * 60)
    
    logit_config = config.get("models", {}).get("logistic_regression", {})
    if not logit_config:
        # Try direct config (for advanced model configs)
        logit_config = config.get("logistic_regression", {})
    
    model = LogisticRegressionModel(**logit_config)
    model.fit(X_train, y_train)
    
    # Save model
    logit_path = artifacts_dir / "logit.pkl"
    model.save(logit_path)
    logger.info(f"Saved logistic regression to {logit_path}")
    
    return model


def train_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
    artifacts_dir: Path,
) -> GradientBoostingModel:
    """Train gradient boosting model."""
    logger.info("\n" + "=" * 60)
    logger.info("Training Gradient Boosting")
    logger.info("=" * 60)
    
    gbm_config = config.get("models", {}).get("gradient_boosting", {})
    if not gbm_config:
        # Try direct config (for advanced model configs)
        gbm_config = config.get("gradient_boosting", {})
    
    model = GradientBoostingModel(**gbm_config)
    model.fit(X_train, y_train)
    
    # Save GBM model
    gbm_path = artifacts_dir / "gbm.pkl"
    model.save(gbm_path)
    logger.info(f"Saved gradient boosting to {gbm_path}")
    
    return model


def train_ft_transformer(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
    artifacts_dir: Path,
) -> FTTransformerModel:
    """Train FT-Transformer model."""
    logger.info("\n" + "=" * 60)
    logger.info("Training FT-Transformer")
    logger.info("=" * 60)
    
    arch = config.get('architecture', {})
    train_cfg = config.get('training', {})
    
    # Ensure numeric values are floats
    learning_rate = train_cfg.get('learning_rate', 1e-4)
    if isinstance(learning_rate, str):
        learning_rate = float(learning_rate)
    
    weight_decay = train_cfg.get('weight_decay', 1e-5)
    if isinstance(weight_decay, str):
        weight_decay = float(weight_decay)
    
    model = FTTransformerModel(
        d_model=arch.get('d_model', 64),
        n_heads=arch.get('n_heads', 4),
        n_layers=arch.get('n_layers', 3),
        d_ff=arch.get('d_ff', 256),
        dropout=arch.get('dropout', 0.1),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=train_cfg.get('batch_size', 64),
        epochs=train_cfg.get('epochs', 100),
        patience=train_cfg.get('patience', 15),
        random_state=config.get('random_state', 42),
    )
    
    model.fit(X_train, y_train, X_val, y_val)
    
    # Save model
    output_path = artifacts_dir / "ft_transformer.pkl"
    model.save(output_path)
    logger.info(f"Model saved to {output_path}")
    
    return model


def train_tabnet(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
    artifacts_dir: Path,
) -> TabNetModel:
    """Train TabNet model."""
    logger.info("\n" + "=" * 60)
    logger.info("Training TabNet")
    logger.info("=" * 60)
    
    arch = config.get('architecture', {})
    train_cfg = config.get('training', {})
    
    # Ensure numeric values are floats
    learning_rate = train_cfg.get('learning_rate', 2e-2)
    if isinstance(learning_rate, str):
        learning_rate = float(learning_rate)
    
    momentum = train_cfg.get('momentum', 0.02)
    if isinstance(momentum, str):
        momentum = float(momentum)
    
    lambda_sparse = arch.get('lambda_sparse', 1e-3)
    if isinstance(lambda_sparse, str):
        lambda_sparse = float(lambda_sparse)
    
    model = TabNetModel(
        n_d=arch.get('n_d', 8),
        n_a=arch.get('n_a', 8),
        n_steps=arch.get('n_steps', 3),
        gamma=arch.get('gamma', 1.3),
        n_independent=arch.get('n_independent', 1),
        n_shared=arch.get('n_shared', 1),
        lambda_sparse=lambda_sparse,
        momentum=momentum,
        learning_rate=learning_rate,
        batch_size=train_cfg.get('batch_size', 256),
        epochs=train_cfg.get('epochs', 100),
        patience=train_cfg.get('patience', 15),
        random_state=config.get('random_state', 42),
    )
    
    model.fit(X_train, y_train, X_val, y_val)
    
    # Save model
    output_path = artifacts_dir / "tabnet.pkl"
    model.save(output_path)
    logger.info(f"Model saved to {output_path}")
    
    return model


def train_stacking_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
    artifacts_dir: Path,
) -> StackingEnsemble:
    """Train stacking ensemble with multiple base models."""
    logger.info("\n" + "=" * 60)
    logger.info("Training Stacking Ensemble")
    logger.info("=" * 60)
    
    # Train or load base models
    base_models = {}
    base_configs = config.get('base_models', {})
    
    for name, model_cfg in base_configs.items():
        model_type = model_cfg.get('type')
        artifact_path = model_cfg.get('artifact')
        
        logger.info(f"\nPreparing base model: {name} ({model_type})")
        
        if artifact_path and Path(artifact_path).exists():
            # Load existing model
            logger.info(f"  Loading from {artifact_path}")
            if model_type == 'logistic_regression':
                base_models[name] = LogisticRegressionModel.load(artifact_path)
            elif model_type == 'gradient_boosting':
                base_models[name] = GradientBoostingModel.load(artifact_path)
            elif model_type == 'ft_transformer':
                base_models[name] = FTTransformerModel.load(artifact_path)
            elif model_type == 'tabnet':
                base_models[name] = TabNetModel.load(artifact_path)
        else:
            # Train new model
            logger.info(f"  Training new {model_type} model")
            model_config_path = model_cfg.get('config')
            if model_config_path:
                model_config = load_config(Path(model_config_path))
            else:
                model_config = {}
            
            if model_type == 'logistic_regression':
                model = train_logistic_regression(X_train, y_train, model_config, artifacts_dir)
                base_models[name] = model
            elif model_type == 'gradient_boosting':
                model = train_gradient_boosting(X_train, y_train, model_config, artifacts_dir)
                base_models[name] = model
            elif model_type == 'ft_transformer':
                model = train_ft_transformer(X_train, y_train, X_val, y_val, model_config, artifacts_dir)
                base_models[name] = model
            elif model_type == 'tabnet':
                model = train_tabnet(X_train, y_train, X_val, y_val, model_config, artifacts_dir)
                base_models[name] = model
    
    # Create and train ensemble
    meta_cfg = config.get('meta_model', {})
    stack_cfg = config.get('stacking', {})
    
    # VALIDATION: Ensure train/val don't overlap (unless validation is intentional subset)
    # Note: Validation can be a subset of training (e.g., 2024 in both train and val)
    # This is allowed for proper validation - the key is that validation is not used for training
    # We only check that indices don't overlap if they're from different time periods
    # For now, we allow overlap since validation is used only for evaluation/calibration, not training
    
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
    
    # CRITICAL FIX: Fit meta-model on TRAINING data only
    # Base models are already trained on X_train, y_train
    # Meta-model learns to combine base predictions using training data
    # Validation data is used ONLY for evaluation, never for training
    logger.info("\nFitting meta-model on training data (base model predictions from X_train)")
    ensemble.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    
    # Evaluate on validation data separately (for logging/metrics only)
    logger.info("\nEvaluating ensemble on validation set")
    val_predictions = ensemble.predict_proba(X_val)
    val_accuracy = (val_predictions >= 0.5) == y_val
    logger.info(f"  Validation accuracy: {val_accuracy.mean():.2%} ({val_accuracy.sum()}/{len(y_val)})")
    
    # Save ensemble
    output_path = artifacts_dir / "ensemble_v1.pkl"
    ensemble.save(output_path)
    logger.info(f"Ensemble saved to {output_path}")
    
    return ensemble


def train_model(
    model_type: Optional[str] = None,
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    config: Optional[dict] = None,
    config_path: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> BaseModel:
    """
    Unified training function for all model types.
    
    Model type can be specified via:
    1. model_type parameter (explicit)
    2. config['model_type'] (from config file)
    3. CLI argument --model
    
    All models inherit from BaseModel interface and follow consistent training pattern.
    
    Args:
        model_type: Model type ("lr", "gbm", "ft_transformer", "tabnet", "stacking_ensemble")
                   If None, will be read from config
        X_train: Training features (required if model_type provided)
        y_train: Training target (required if model_type provided)
        X_val: Validation features (required for advanced models)
        y_val: Validation target (required for advanced models)
        config: Model configuration dict (if None, will be loaded from config_path)
        config_path: Path to config file (used if config is None)
        artifacts_dir: Directory to save model artifacts
    
    Returns:
        Trained model instance (inherits from BaseModel)
    
    Raises:
        ValueError: If model_type cannot be determined or is invalid
    """
    # Load config if not provided
    if config is None:
        if config_path:
            config = load_config(config_path)
        else:
            config = {}
    
    # Determine model type from config if not provided
    if model_type is None:
        model_type = config.get('model_type')
        if model_type is None:
            raise ValueError(
                "model_type must be provided either as parameter, CLI argument, "
                "or in config file as 'model_type'"
            )
        logger.info(f"Model type read from config: {model_type}")
    
    # Normalize model_type to handle variations
    model_type_normalized = model_type.lower().replace("_", "").replace("-", "")
    
    # Set up artifacts directory
    if artifacts_dir is None:
        artifacts_dir = (
            Path(__file__).parent.parent.parent
            / "models"
            / "artifacts"
            / f"nfl_{model_type_normalized}"
        )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Training model type: {model_type} (normalized: {model_type_normalized})")
    
    # Route to appropriate training function
    if model_type_normalized in ["lr", "logistic", "logisticregression"]:
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train are required for logistic regression")
        return train_logistic_regression(X_train, y_train, config, artifacts_dir)
        
    elif model_type_normalized in ["gbm", "gradientboosting", "xgboost"]:
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train are required for gradient boosting")
        return train_gradient_boosting(X_train, y_train, config, artifacts_dir)
        
    elif model_type_normalized in ["fttransformer", "ft_transformer", "transformer"]:
        if X_train is None or y_train is None or X_val is None or y_val is None:
            raise ValueError("X_train, y_train, X_val, and y_val are required for FT-Transformer")
        return train_ft_transformer(X_train, y_train, X_val, y_val, config, artifacts_dir)
        
    elif model_type_normalized in ["tabnet"]:
        if X_train is None or y_train is None or X_val is None or y_val is None:
            raise ValueError("X_train, y_train, X_val, and y_val are required for TabNet")
        return train_tabnet(X_train, y_train, X_val, y_val, config, artifacts_dir)
        
    elif model_type_normalized in ["stackingensemble", "stacking_ensemble", "ensemble"]:
        if X_train is None or y_train is None or X_val is None or y_val is None:
            raise ValueError("X_train, y_train, X_val, and y_val are required for stacking ensemble")
        return train_stacking_ensemble(X_train, y_train, X_val, y_val, config, artifacts_dir)
        
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported types: lr, gbm, ft_transformer, tabnet, stacking_ensemble"
        )


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: Optional[dict] = None,
    artifacts_dir: Optional[Path] = None,
) -> Tuple[LogisticRegressionModel, GradientBoostingModel, EnsembleModel]:
    """
    Train logistic regression, GBM, and simple ensemble models.
    
    This function maintains backward compatibility with existing baseline training.
    For advanced models, use train_model() instead.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features (for tuning)
        y_val: Validation target
        config: Model configuration dict
        artifacts_dir: Directory to save model artifacts
    
    Returns:
        Tuple of (logit_model, gbm_model, ensemble_model)
    """
    if config is None:
        config = load_config()
    
    if artifacts_dir is None:
        artifacts_dir = (
            Path(__file__).parent.parent.parent
            / "models"
            / "artifacts"
            / "nfl_baseline"
        )
    
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Train logistic regression
    logit_model = train_logistic_regression(X_train, y_train, config, artifacts_dir)
    
    # Train gradient boosting
    gbm_model = train_gradient_boosting(X_train, y_train, config, artifacts_dir)
    
    # Tune ensemble weight on validation set
    logger.info("\n" + "=" * 60)
    logger.info("Tuning Ensemble Weight")
    logger.info("=" * 60)
    
    from eval.metrics import brier_score
    
    best_weight = config.get("models", {}).get("ensemble", {}).get("weight", 0.7)
    best_brier = float("inf")
    
    # Try different weights
    weights_to_try = [0.5, 0.6, 0.7, 0.8, 0.9]
    for weight in weights_to_try:
        ensemble = EnsembleModel(logit_model, gbm_model, weight=weight)
        p_val = ensemble.predict_proba(X_val)
        brier = brier_score(y_val.values, p_val)
        
        logger.info(f"Weight {weight:.2f}: Brier = {brier:.4f}")
        
        if brier < best_brier:
            best_brier = brier
            best_weight = weight
    
    logger.info(f"Best ensemble weight: {best_weight:.2f} (Brier: {best_brier:.4f})")
    
    # Create final ensemble with best weight
    ensemble_model = EnsembleModel(logit_model, gbm_model, weight=best_weight)
    
    # Save ensemble config
    ensemble_config = {"weight": best_weight}
    ensemble_config_path = artifacts_dir / "ensemble.json"
    with open(ensemble_config_path, "w") as f:
        json.dump(ensemble_config, f)
    logger.info(f"Saved ensemble config to {ensemble_config_path}")
    
    return logit_model, gbm_model, ensemble_model


def apply_calibration(
    model: BaseModel,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
) -> Union[BaseModel, CalibratedModel]:
    """
    Apply calibration to a model if enabled in config.
    
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation target
        config: Configuration dict with calibration settings
    
    Returns:
        Calibrated model if enabled, otherwise original model
    """
    cal_config = config.get('calibration', {})
    
    if not cal_config.get('enabled', False):
        return model
    
    method = cal_config.get('method', 'platt')
    logger.info(f"Applying {method} calibration")
    
    calibrated = CalibratedModel(base_model=model, method=method)
    calibrated.fit_calibration(X_val, y_val)
    
    return calibrated


def run_training_pipeline(
    features_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    backtest_config_path: Optional[Path] = None,
) -> Tuple[
    LogisticRegressionModel,
    GradientBoostingModel,
    EnsembleModel,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
]:
    """
    Run complete baseline training pipeline.
    
    This function maintains backward compatibility with existing baseline training.
    For advanced models, use run_advanced_training_pipeline() or train_model() directly.
    
    Returns:
        Tuple of (logit_model, gbm_model, ensemble_model, X_train, y_train, X_val, y_val, X_test, y_test, df_full)
        df_full is the full dataframe with all columns (for downstream use in evaluation).
    """
    logger.info("=" * 60)
    logger.info("NFL Baseline Model Training Pipeline")
    logger.info("=" * 60)
    
    # Load configs
    config = load_config(config_path) if config_path else load_config()
    backtest_config = load_backtest_config() if backtest_config_path is None else yaml.safe_load(open(backtest_config_path))
    
    # Get feature table from config
    feature_table = backtest_config.get("feature_table", "baseline")
    
    # Load features (single load - returns full df to avoid reloading)
    X, y, feature_cols, df = load_features(features_path, feature_table=feature_table)
    
    # Split data
    train_seasons = backtest_config["splits"]["train_seasons"]
    validation_season = backtest_config["splits"]["validation_season"]
    test_season = backtest_config["splits"]["test_season"]
    
    X_train, y_train, X_val, y_val, X_test, y_test = split_by_season(
        X, y, df, train_seasons, validation_season, test_season
    )
    
    # Train models
    logit_model, gbm_model, ensemble_model = train_models(
        X_train, y_train, X_val, y_val, config
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    
    return logit_model, gbm_model, ensemble_model, X_train, y_train, X_val, y_val, X_test, y_test, df


def run_advanced_training_pipeline(
    model_type: Optional[str] = None,
    config_path: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
    feature_table: Optional[str] = None,
    apply_calibration_flag: bool = True,
) -> Tuple[BaseModel, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Run advanced model training pipeline.
    
    Model type can be specified via:
    1. model_type parameter (explicit)
    2. config['model_type'] (from config file)
    3. CLI argument --model
    
    Args:
        model_type: Model type ("ft_transformer", "tabnet", "stacking_ensemble", "lr", "gbm")
                   If None, will be read from config file
        config_path: Path to model config file (model_type can be specified in config)
        artifacts_dir: Output directory for artifacts
        feature_table: Feature table name (if None, uses config)
        apply_calibration_flag: Whether to apply calibration
    
    Returns:
        Tuple of (model, X_train, y_train, X_val, y_val, X_test, y_test)
    """
    logger.info("=" * 60)
    logger.info("NFL Model Training Pipeline")
    logger.info("=" * 60)
    
    # Load config
    config = {}
    if config_path:
        config = load_config(config_path)
        logger.info(f"Loaded config from: {config_path}")
    
    # Determine model type: parameter > config file > default
    if model_type is None:
        model_type = config.get('model_type')
        if model_type:
            logger.info(f"Model type read from config: {model_type}")
    
    # If still no model type, use defaults based on config file name
    if model_type is None:
        if config_path:
            # Infer from config filename
            config_name = config_path.stem.lower()
            if 'ft_transformer' in config_name or 'transformer' in config_name:
                model_type = 'ft_transformer'
            elif 'tabnet' in config_name:
                model_type = 'tabnet'
            elif 'ensemble' in config_name:
                model_type = 'stacking_ensemble'
            else:
                # Default configs
                default_configs = {
                    'ft_transformer': 'config/models/nfl_ft_transformer.yaml',
                    'tabnet': 'config/models/nfl_tabnet.yaml',
                    'stacking_ensemble': 'config/models/nfl_ensemble_v1.yaml',
                    'ensemble': 'config/models/nfl_ensemble_v1.yaml',
                }
                # Try to infer from config path
                for mt, default_path in default_configs.items():
                    if str(config_path) == default_path or default_path in str(config_path):
                        model_type = mt
                        break
        else:
            raise ValueError(
                "model_type must be provided either as parameter, CLI argument, "
                "or in config file as 'model_type' field"
            )
    
    # Normalize ensemble -> stacking_ensemble
    if model_type == 'ensemble':
        model_type = 'stacking_ensemble'
    
    logger.info(f"Training model type: {model_type}")
    
    # Set up artifacts directory
    if artifacts_dir is None:
        artifacts_dir = Path(__file__).parent.parent.parent / "models" / "artifacts" / f"nfl_{model_type}"
    
    # Load data
    logger.info("Loading feature data...")
    backtest_config = load_backtest_config()
    
    if feature_table is None:
        feature_table = config.get('features', {}).get('feature_table', 'baseline')
    
    X, y, feature_cols, df = load_features(feature_table=feature_table)
    
    # Apply feature selection if specified in config
    if config.get('features', {}).get('feature_groups'):
        feature_cols = get_feature_columns(df, config)
        X = X[feature_cols]
    
    # Split data
    train_seasons = backtest_config['splits']['train_seasons']
    val_season = backtest_config['splits']['validation_season']
    test_season = backtest_config['splits']['test_season']
    
    X_train, y_train, X_val, y_val, X_test, y_test = split_by_season(
        X, y, df, train_seasons, val_season, test_season
    )
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train model using unified train_model() function
    # train_model() will handle model_type parsing from config if needed
    model = train_model(
        model_type=model_type,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        config=config,
        config_path=config_path,
        artifacts_dir=artifacts_dir,
    )
    
    # Verify model implements BaseModel interface
    if not isinstance(model, BaseModel):
        raise TypeError(f"Model {type(model)} does not implement BaseModel interface")
    
    # Apply calibration if enabled
    if apply_calibration_flag:
        model = apply_calibration(model, X_val, y_val, config)
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    
    return model, X_train, y_train, X_val, y_val, X_test, y_test


def main():
    """
    CLI entrypoint for unified trainer.
    
    Supports training all model types: lr, gbm, ft_transformer, tabnet, stacking_ensemble.
    Model type can be specified via --model argument or read from config file.
    
    Examples:
        # Train baseline models (backward compatible)
        python -m models.training.trainer
        
        # Train FT-Transformer
        python -m models.training.trainer --model ft_transformer
        
        # Train TabNet
        python -m models.training.trainer --model tabnet
        
        # Train Stacking Ensemble
        python -m models.training.trainer --model stacking_ensemble
        
        # Train with custom config (model_type read from config)
        python -m models.training.trainer --config config/models/nfl_ft_transformer.yaml
        
        # Train with custom config and explicit model type
        python -m models.training.trainer --model ft_transformer --config config/models/nfl_ft_transformer.yaml
    """
    parser = argparse.ArgumentParser(
        description="Train NFL prediction models (unified trainer for all model types)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train baseline models (backward compatible)
  python -m models.training.trainer
  
  # Train FT-Transformer
  python -m models.training.trainer --model ft_transformer
  
  # Train TabNet
  python -m models.training.trainer --model tabnet
  
  # Train Stacking Ensemble
  python -m models.training.trainer --model stacking_ensemble
  
  # Train with config (model_type read from config file)
  python -m models.training.trainer --config config/models/nfl_ft_transformer.yaml
  
  # Train with explicit model type and config
  python -m models.training.trainer --model ft_transformer --config config/models/nfl_ft_transformer.yaml
        """
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['lr', 'gbm', 'ft_transformer', 'tabnet', 'stacking_ensemble', 'ensemble'],
        default=None,
        help='Model type to train. If not specified, will be read from config file or defaults to baseline models.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to model config file. Model type can be specified in config as "model_type" field.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for model artifacts'
    )
    parser.add_argument(
        '--feature-table',
        type=str,
        default=None,
        help='Feature table name (baseline, phase2, phase2b)'
    )
    parser.add_argument(
        '--no-calibration',
        action='store_true',
        help='Skip calibration (advanced models only)'
    )
    
    args = parser.parse_args()
    
    # If no model specified and no config, run baseline pipeline (backward compatible)
    if args.model is None and args.config is None:
        logger.info("No model or config specified, running baseline training pipeline...")
        run_training_pipeline()
        return
    
    # Determine model type: CLI argument > config file > error
    model_type = args.model
    config_path = Path(args.config) if args.config else None
    
    # Load config if provided
    if config_path:
        config = load_config(config_path)
        # If model_type not in CLI args, try to read from config
        if model_type is None:
            model_type = config.get('model_type')
            if model_type:
                logger.info(f"Model type read from config file: {model_type}")
    else:
        config = {}
    
    # If still no model type, use defaults based on config file name
    if model_type is None:
        if config_path:
            # Infer from config filename
            config_name = config_path.stem.lower()
            if 'ft_transformer' in config_name or 'transformer' in config_name:
                model_type = 'ft_transformer'
            elif 'tabnet' in config_name:
                model_type = 'tabnet'
            elif 'ensemble' in config_name:
                model_type = 'stacking_ensemble'
            elif 'baseline' in config_name:
                # Default to baseline models
                logger.info("Baseline config detected, running baseline training pipeline...")
                run_training_pipeline(config_path=config_path)
                return
            else:
                raise ValueError(
                    f"Could not determine model type. "
                    f"Please specify --model argument or include 'model_type' in config file."
                )
        else:
            raise ValueError(
                "Model type must be specified via --model argument or config file."
            )
    
    # Normalize ensemble -> stacking_ensemble
    if model_type == 'ensemble':
        model_type = 'stacking_ensemble'
    
    # Run unified training pipeline
    artifacts_dir = Path(args.output_dir) if args.output_dir else None
    
    model, X_train, y_train, X_val, y_val, X_test, y_test = run_advanced_training_pipeline(
        model_type=model_type,
        config_path=config_path,
        artifacts_dir=artifacts_dir,
        feature_table=args.feature_table,
        apply_calibration_flag=not args.no_calibration,
    )
    
    # Evaluate on test set
    logger.info("\n" + "=" * 60)
    logger.info("Test Set Evaluation")
    logger.info("=" * 60)
    
    y_pred_proba = model.predict_proba(X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    accuracy = (y_pred == y_test).mean()
    brier = ((y_pred_proba - y_test) ** 2).mean()
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Brier Score: {brier:.4f}")
    
    if isinstance(model, CalibratedModel):
        logger.info("Model is calibrated")
    
    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()
