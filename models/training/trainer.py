"""
Model Trainer

Handles data loading, splitting, and training of baseline models.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Tuple, List, Optional
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.architectures.logistic_regression import LogisticRegressionModel
from models.architectures.gradient_boosting import GradientBoostingModel
from models.architectures.ensemble import EnsembleModel
from features.feature_table_registry import get_feature_table_path, validate_feature_table_exists

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load model configuration."""
    config_path = Path(__file__).parent.parent.parent / "config" / "models" / "nfl_baseline.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: Optional[dict] = None,
    artifacts_dir: Optional[Path] = None,
) -> Tuple[LogisticRegressionModel, GradientBoostingModel, EnsembleModel]:
    """
    Train logistic regression, GBM, and ensemble models.
    
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
    logger.info("\n" + "=" * 60)
    logger.info("Training Logistic Regression")
    logger.info("=" * 60)
    logit_config = config["models"]["logistic_regression"]
    logit_model = LogisticRegressionModel(**logit_config)
    logit_model.fit(X_train, y_train)
    
    # Save logit model
    logit_path = artifacts_dir / "logit.pkl"
    logit_model.save(logit_path)
    logger.info(f"Saved logistic regression to {logit_path}")
    
    # Train gradient boosting
    logger.info("\n" + "=" * 60)
    logger.info("Training Gradient Boosting")
    logger.info("=" * 60)
    gbm_config = config["models"]["gradient_boosting"]
    gbm_model = GradientBoostingModel(**gbm_config)
    gbm_model.fit(X_train, y_train)
    
    # Save GBM model
    gbm_path = artifacts_dir / "gbm.pkl"
    gbm_model.save(gbm_path)
    logger.info(f"Saved gradient boosting to {gbm_path}")
    
    # Tune ensemble weight on validation set
    logger.info("\n" + "=" * 60)
    logger.info("Tuning Ensemble Weight")
    logger.info("=" * 60)
    
    from eval.metrics import brier_score
    
    best_weight = config["models"]["ensemble"]["weight"]
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
    import json
    ensemble_config = {"weight": best_weight}
    ensemble_config_path = artifacts_dir / "ensemble.json"
    with open(ensemble_config_path, "w") as f:
        json.dump(ensemble_config, f)
    logger.info(f"Saved ensemble config to {ensemble_config_path}")
    
    return logit_model, gbm_model, ensemble_model


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
    Run complete training pipeline.
    
    Returns:
        Tuple of (logit_model, gbm_model, ensemble_model, X_train, y_train, X_val, y_val, X_test, y_test, df_full)
        df_full is the full dataframe with all columns (for downstream use in evaluation).
    """
    logger.info("=" * 60)
    logger.info("NFL Baseline Model Training Pipeline")
    logger.info("=" * 60)
    
    # Load configs
    config = load_config() if config_path is None else yaml.safe_load(open(config_path))
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


if __name__ == "__main__":
    run_training_pipeline()

