"""
Advanced Model Training Script

Trains FT-Transformer, TabNet, or Stacking Ensemble models
using configuration files.

Usage:
    # Train FT-Transformer
    python -m models.training.train_advanced --model ft_transformer

    # Train TabNet
    python -m models.training.train_advanced --model tabnet

    # Train Stacking Ensemble (combines all models)
    python -m models.training.train_advanced --model ensemble

    # Train with specific config
    python -m models.training.train_advanced --model ft_transformer --config config/models/nfl_ft_transformer.yaml
"""

import sys
from pathlib import Path
import argparse
import yaml
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np

from models.architectures.ft_transformer import FTTransformerModel
from models.architectures.tabnet import TabNetModel
from models.architectures.stacking_ensemble import StackingEnsemble
from models.architectures.logistic_regression import LogisticRegressionModel
from models.architectures.gradient_boosting import GradientBoostingModel
from models.calibration import CalibratedModel
from models.training.trainer import load_features, split_by_season, load_backtest_config
from features.registry import FeatureRegistry, FeatureGroup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model_config(config_path: Path) -> dict:
    """Load model configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_feature_columns(df: pd.DataFrame, config: dict) -> list:
    """Get feature columns based on config."""
    all_columns = list(df.columns)

    # Get groups from config
    groups = config.get('features', {}).get('feature_groups')

    if groups:
        group_enums = [FeatureGroup(g) for g in groups]
        allowed = set(FeatureRegistry.get_feature_groups(group_enums))
        columns = [c for c in all_columns if c in allowed]
    else:
        # Use default exclusions
        exclude = FeatureRegistry.get_exclude_columns()
        columns = [c for c in all_columns if c not in exclude]

    return columns


def train_ft_transformer(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
    output_dir: Path,
) -> FTTransformerModel:
    """Train FT-Transformer model."""
    logger.info("=" * 60)
    logger.info("Training FT-Transformer")
    logger.info("=" * 60)

    arch = config.get('architecture', {})
    train_cfg = config.get('training', {})

    model = FTTransformerModel(
        d_model=arch.get('d_model', 64),
        n_heads=arch.get('n_heads', 4),
        n_layers=arch.get('n_layers', 3),
        d_ff=arch.get('d_ff', 256),
        dropout=arch.get('dropout', 0.1),
        learning_rate=train_cfg.get('learning_rate', 1e-4),
        weight_decay=train_cfg.get('weight_decay', 1e-5),
        batch_size=train_cfg.get('batch_size', 64),
        epochs=train_cfg.get('epochs', 100),
        patience=train_cfg.get('patience', 15),
        random_state=config.get('random_state', 42),
    )

    model.fit(X_train, y_train, X_val, y_val)

    # Save model
    output_path = output_dir / "ft_transformer.pkl"
    model.save(output_path)
    logger.info(f"Model saved to {output_path}")

    return model


def train_tabnet(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
    output_dir: Path,
) -> TabNetModel:
    """Train TabNet model."""
    logger.info("=" * 60)
    logger.info("Training TabNet")
    logger.info("=" * 60)

    arch = config.get('architecture', {})
    train_cfg = config.get('training', {})

    model = TabNetModel(
        n_d=arch.get('n_d', 8),
        n_a=arch.get('n_a', 8),
        n_steps=arch.get('n_steps', 3),
        gamma=arch.get('gamma', 1.3),
        n_independent=arch.get('n_independent', 1),
        n_shared=arch.get('n_shared', 1),
        lambda_sparse=arch.get('lambda_sparse', 1e-3),
        momentum=train_cfg.get('momentum', 0.02),
        learning_rate=train_cfg.get('learning_rate', 2e-2),
        batch_size=train_cfg.get('batch_size', 256),
        epochs=train_cfg.get('epochs', 100),
        patience=train_cfg.get('patience', 15),
        random_state=config.get('random_state', 42),
    )

    model.fit(X_train, y_train, X_val, y_val)

    # Save model
    output_path = output_dir / "tabnet.pkl"
    model.save(output_path)
    logger.info(f"Model saved to {output_path}")

    return model


def train_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
    output_dir: Path,
) -> StackingEnsemble:
    """Train stacking ensemble with multiple base models."""
    logger.info("=" * 60)
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
            if model_type == 'logistic_regression':
                model = LogisticRegressionModel()
                model.fit(X_train, y_train)
                base_models[name] = model
            elif model_type == 'gradient_boosting':
                model = GradientBoostingModel()
                model.fit(X_train, y_train)
                base_models[name] = model
            elif model_type == 'ft_transformer':
                ft_config_path = model_cfg.get('config')
                if ft_config_path:
                    ft_config = load_model_config(ft_config_path)
                else:
                    ft_config = {}
                model = train_ft_transformer(X_train, y_train, X_val, y_val, ft_config, output_dir)
                base_models[name] = model
            elif model_type == 'tabnet':
                tn_config_path = model_cfg.get('config')
                if tn_config_path:
                    tn_config = load_model_config(tn_config_path)
                else:
                    tn_config = {}
                model = train_tabnet(X_train, y_train, X_val, y_val, tn_config, output_dir)
                base_models[name] = model

    # Create and train ensemble
    meta_cfg = config.get('meta_model', {})
    stack_cfg = config.get('stacking', {})

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

    # Fit meta-model on validation data
    logger.info("\nFitting meta-model on validation data")
    ensemble.fit(X_val, y_val)

    # Save ensemble
    output_path = output_dir / "ensemble_v1.pkl"
    ensemble.save(output_path)
    logger.info(f"Ensemble saved to {output_path}")

    return ensemble


def apply_calibration(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
) -> CalibratedModel:
    """Apply calibration to a model."""
    cal_config = config.get('calibration', {})

    if not cal_config.get('enabled', True):
        return model

    method = cal_config.get('method', 'platt')
    logger.info(f"Applying {method} calibration")

    calibrated = CalibratedModel(base_model=model, method=method)
    calibrated.fit_calibration(X_val, y_val)

    return calibrated


def main():
    parser = argparse.ArgumentParser(description="Train advanced NFL prediction models")
    parser.add_argument(
        '--model',
        type=str,
        choices=['ft_transformer', 'tabnet', 'ensemble'],
        required=True,
        help='Model type to train'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to model config file (uses defaults if not specified)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/artifacts/nfl_advanced',
        help='Output directory for model artifacts'
    )

    args = parser.parse_args()

    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    if args.config:
        config = load_model_config(Path(args.config))
    else:
        # Use default config
        default_configs = {
            'ft_transformer': 'config/models/nfl_ft_transformer.yaml',
            'tabnet': 'config/models/nfl_tabnet.yaml',
            'ensemble': 'config/models/nfl_ensemble_v1.yaml',
        }
        config_path = Path(default_configs[args.model])
        if config_path.exists():
            config = load_model_config(config_path)
        else:
            config = {}

    # Load data
    logger.info("Loading feature data...")
    backtest_config = load_backtest_config()
    feature_table = config.get('features', {}).get('feature_table', 'baseline')

    X, y, feature_cols, df = load_features(feature_table=feature_table)

    # Split data
    train_seasons = backtest_config['splits']['train_seasons']
    val_season = backtest_config['splits']['validation_season']
    test_season = backtest_config['splits']['test_season']

    X_train, y_train, X_val, y_val, X_test, y_test = split_by_season(
        X, y, df, train_seasons, val_season, test_season
    )

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train model
    if args.model == 'ft_transformer':
        model = train_ft_transformer(X_train, y_train, X_val, y_val, config, output_dir)
    elif args.model == 'tabnet':
        model = train_tabnet(X_train, y_train, X_val, y_val, config, output_dir)
    elif args.model == 'ensemble':
        model = train_ensemble(X_train, y_train, X_val, y_val, config, output_dir)

    # Apply calibration
    calibrated_model = apply_calibration(model, X_val, y_val, config)

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

    if isinstance(calibrated_model, CalibratedModel):
        y_pred_cal = calibrated_model.predict_proba(X_test)
        brier_cal = ((y_pred_cal - y_test) ** 2).mean()
        logger.info(f"Brier Score (calibrated): {brier_cal:.4f}")

    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()
