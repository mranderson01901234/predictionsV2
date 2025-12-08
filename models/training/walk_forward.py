"""
Walk-Forward Validation Framework

Implements walk-forward (time-series) cross-validation for NFL prediction models.
This simulates real deployment by training on all available historical data
at each point in time, then testing on the next season.

Walk-Forward Splits:
  Split 1: Train 2015-2018, Test 2019
  Split 2: Train 2015-2019, Test 2020
  Split 3: Train 2015-2020, Test 2021
  Split 4: Train 2015-2021, Test 2022
  Split 5: Train 2015-2022, Test 2023
  Split 6: Train 2015-2023, Test 2024

Benefits:
- More robust performance estimates (multiple test sets)
- Simulates actual deployment conditions
- Detects regime changes over time
- Provides variance estimates for metrics

Usage:
    python -m models.training.walk_forward \
        --features data/nfl/processed/game_features_phase3.parquet \
        --model ensemble \
        --output results/walk_forward/
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from pathlib import Path
import sys
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardSplit:
    """Single walk-forward split configuration."""
    name: str
    train_seasons: List[int]
    test_season: int

    @property
    def train_start(self) -> int:
        return min(self.train_seasons)

    @property
    def train_end(self) -> int:
        return max(self.train_seasons)


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward split."""
    split_name: str
    train_seasons: List[int]
    test_season: int
    n_train_games: int
    n_test_games: int

    # Core metrics
    accuracy: float
    brier_score: float
    log_loss_value: float

    # ROI metrics
    roi_flat: float
    roi_kelly: float
    roi_threshold_52: float
    roi_threshold_55: float

    # Calibration metrics
    calibration_error: float
    confidence_50_60_accuracy: float  # Critical tier

    # Predictions for further analysis
    predictions: pd.DataFrame = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding predictions)."""
        return {
            'split_name': self.split_name,
            'train_start': min(self.train_seasons),
            'train_end': max(self.train_seasons),
            'test_season': self.test_season,
            'n_train': self.n_train_games,
            'n_test': self.n_test_games,
            'accuracy': self.accuracy,
            'brier': self.brier_score,
            'log_loss': self.log_loss_value,
            'roi_flat': self.roi_flat,
            'roi_kelly': self.roi_kelly,
            'roi_52': self.roi_threshold_52,
            'roi_55': self.roi_threshold_55,
            'cal_error': self.calibration_error,
            'conf_50_60_acc': self.confidence_50_60_accuracy,
        }


class WalkForwardValidator:
    """
    Walk-forward validation for NFL prediction model.

    Simulates real deployment by training on all available historical data
    at each point in time, then testing on the next season.

    Args:
        features_df: Complete feature dataset with 'season' column
        model_factory: Callable that returns a new model instance
        min_train_seasons: Minimum seasons required for training (default: 4)
        calibration_method: Method for calibration ('platt', 'isotonic', 'temperature')
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        model_factory: callable,
        min_train_seasons: int = 4,
        calibration_method: str = 'isotonic',
        feature_cols: Optional[List[str]] = None,
    ):
        self.features_df = features_df
        self.model_factory = model_factory
        self.min_train_seasons = min_train_seasons
        self.calibration_method = calibration_method

        # Identify feature columns
        if feature_cols is None:
            exclude_cols = [
                'game_id', 'season', 'week', 'date', 'home_team', 'away_team',
                'home_score', 'away_score', 'home_win', 'close_spread', 'close_total',
                'open_spread', 'open_total', 'game_type',
            ]
            feature_cols = [c for c in features_df.columns if c not in exclude_cols]
        self.feature_cols = feature_cols

        # Ensure home_win target exists
        if 'home_win' not in features_df.columns:
            features_df['home_win'] = (features_df['home_score'] > features_df['away_score']).astype(int)

        self.seasons = sorted(features_df['season'].unique())
        self.results: List[WalkForwardResult] = []

    def generate_splits(self, start_test_season: Optional[int] = None) -> List[WalkForwardSplit]:
        """
        Generate walk-forward splits.

        Args:
            start_test_season: First season to use as test set (defaults to min_train_seasons + first_season)

        Returns:
            List of WalkForwardSplit configurations
        """
        splits = []

        for i, test_season in enumerate(self.seasons):
            train_seasons = [s for s in self.seasons if s < test_season]

            # Check minimum training seasons
            if len(train_seasons) < self.min_train_seasons:
                continue

            # Check start_test_season constraint
            if start_test_season and test_season < start_test_season:
                continue

            splits.append(WalkForwardSplit(
                name=f"train_{min(train_seasons)}-{max(train_seasons)}_test_{test_season}",
                train_seasons=train_seasons,
                test_season=test_season
            ))

        return splits

    def run_single_split(
        self,
        split: WalkForwardSplit,
        apply_calibration: bool = True,
    ) -> WalkForwardResult:
        """
        Run training and evaluation for a single split.

        Args:
            split: Walk-forward split configuration
            apply_calibration: Whether to apply calibration

        Returns:
            WalkForwardResult with all metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {split.name}")
        logger.info(f"Train: {split.train_start}-{split.train_end} ({len(split.train_seasons)} seasons)")
        logger.info(f"Test: {split.test_season}")
        logger.info(f"{'='*60}")

        # Split data
        train_mask = self.features_df['season'].isin(split.train_seasons)
        test_mask = self.features_df['season'] == split.test_season

        train_df = self.features_df[train_mask].copy()
        test_df = self.features_df[test_mask].copy()

        # Prepare features
        X_train = train_df[self.feature_cols].fillna(0)
        y_train = train_df['home_win']
        X_test = test_df[self.feature_cols].fillna(0)
        y_test = test_df['home_win']

        n_train = len(X_train)
        n_test = len(X_test)

        logger.info(f"Train size: {n_train} games")
        logger.info(f"Test size: {n_test} games")
        logger.info(f"Features: {len(self.feature_cols)}")

        # Create and train model
        model = self.model_factory()

        # Split training data into train and validation for calibration
        # Use last season of training as validation
        val_season = max(split.train_seasons)
        train_train_mask = train_df['season'] < val_season
        train_val_mask = train_df['season'] == val_season

        X_train_train = train_df.loc[train_train_mask, self.feature_cols].fillna(0)
        y_train_train = train_df.loc[train_train_mask, 'home_win']
        X_train_val = train_df.loc[train_val_mask, self.feature_cols].fillna(0)
        y_train_val = train_df.loc[train_val_mask, 'home_win']

        logger.info(f"Training: {len(X_train_train)} games, Validation: {len(X_train_val)} games")

        # Train model (some models need validation data)
        if hasattr(model, 'fit') and 'X_val' in model.fit.__code__.co_varnames:
            model.fit(X_train_train, y_train_train, X_val=X_train_val, y_val=y_train_val)
        else:
            model.fit(X_train, y_train)

        # Apply calibration
        if apply_calibration:
            from models.calibration import CalibratedModel
            calibrated_model = CalibratedModel(base_model=model, method=self.calibration_method)
            calibrated_model.fit_calibration(X_train_val, y_train_val)
            y_pred_proba = calibrated_model.predict_proba(X_test)
        else:
            y_pred_proba = model.predict_proba(X_test)

        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_pred_proba)
        ll = log_loss(y_test, np.clip(y_pred_proba, 1e-7, 1-1e-7))

        # ROI metrics
        roi_flat = self._compute_roi_flat(y_test.values, y_pred_proba)
        roi_kelly = self._compute_roi_kelly(y_test.values, y_pred_proba)
        roi_52 = self._compute_roi_threshold(y_test.values, y_pred_proba, threshold=0.52)
        roi_55 = self._compute_roi_threshold(y_test.values, y_pred_proba, threshold=0.55)

        # Calibration metrics
        cal_error = self._compute_calibration_error(y_test.values, y_pred_proba)
        conf_50_60_acc = self._compute_confidence_tier_accuracy(y_test.values, y_pred_proba, 0.50, 0.60)

        logger.info(f"\nResults for {split.name}:")
        logger.info(f"  Accuracy: {accuracy:.3f}")
        logger.info(f"  Brier Score: {brier:.4f}")
        logger.info(f"  Log Loss: {ll:.4f}")
        logger.info(f"  ROI (flat): {roi_flat:.2%}")
        logger.info(f"  ROI (52%+ edge): {roi_52:.2%}")
        logger.info(f"  Calibration Error: {cal_error:.4f}")
        logger.info(f"  50-60% Conf Accuracy: {conf_50_60_acc:.3f}")

        # Store predictions
        predictions = test_df[['game_id', 'season', 'week', 'home_team', 'away_team']].copy()
        predictions['y_true'] = y_test.values
        predictions['y_pred_proba'] = y_pred_proba
        predictions['y_pred'] = y_pred
        predictions['correct'] = (y_pred == y_test.values).astype(int)

        return WalkForwardResult(
            split_name=split.name,
            train_seasons=split.train_seasons,
            test_season=split.test_season,
            n_train_games=n_train,
            n_test_games=n_test,
            accuracy=accuracy,
            brier_score=brier,
            log_loss_value=ll,
            roi_flat=roi_flat,
            roi_kelly=roi_kelly,
            roi_threshold_52=roi_52,
            roi_threshold_55=roi_55,
            calibration_error=cal_error,
            confidence_50_60_accuracy=conf_50_60_acc,
            predictions=predictions,
        )

    def run_all_splits(
        self,
        start_test_season: Optional[int] = None,
        apply_calibration: bool = True,
    ) -> List[WalkForwardResult]:
        """
        Run all walk-forward splits.

        Args:
            start_test_season: First season to test (defaults to after min_train_seasons)
            apply_calibration: Whether to apply calibration

        Returns:
            List of WalkForwardResult for each split
        """
        splits = self.generate_splits(start_test_season)
        logger.info(f"Generated {len(splits)} walk-forward splits")

        self.results = []
        for split in splits:
            result = self.run_single_split(split, apply_calibration)
            self.results.append(result)

        return self.results

    def summarize_results(self) -> pd.DataFrame:
        """
        Summarize results across all splits.

        Returns:
            DataFrame with summary statistics
        """
        if not self.results:
            raise ValueError("No results available. Run run_all_splits() first.")

        summary_data = [r.to_dict() for r in self.results]
        summary = pd.DataFrame(summary_data)

        # Add aggregate statistics
        logger.info("\n" + "="*60)
        logger.info("WALK-FORWARD VALIDATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Number of splits: {len(self.results)}")
        logger.info(f"Total test games: {summary['n_test'].sum()}")
        logger.info(f"\nMean Accuracy: {summary['accuracy'].mean():.3f} "
                   f"(std: {summary['accuracy'].std():.3f})")
        logger.info(f"Mean Brier: {summary['brier'].mean():.4f}")
        logger.info(f"Mean Log Loss: {summary['log_loss'].mean():.4f}")
        logger.info(f"Mean ROI (flat): {summary['roi_flat'].mean():.2%}")
        logger.info(f"Mean ROI (52%+ edge): {summary['roi_52'].mean():.2%}")
        logger.info(f"Mean Calibration Error: {summary['cal_error'].mean():.4f}")
        logger.info(f"Mean 50-60% Conf Accuracy: {summary['conf_50_60_acc'].mean():.3f}")

        return summary

    def _compute_roi_flat(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Compute ROI with flat betting on all games."""
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Assuming -110 odds on both sides
        wins = (y_pred == y_true).sum()
        losses = len(y_true) - wins

        profit = wins * 100 - losses * 110
        wagered = len(y_true) * 110

        return profit / wagered if wagered > 0 else 0.0

    def _compute_roi_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.52
    ) -> float:
        """Compute ROI with betting only when edge exceeds threshold."""
        edge = np.abs(y_pred_proba - 0.5)
        mask = edge >= (threshold - 0.5)

        if mask.sum() == 0:
            return 0.0

        y_true_filtered = y_true[mask]
        y_pred_filtered = (y_pred_proba[mask] >= 0.5).astype(int)

        wins = (y_pred_filtered == y_true_filtered).sum()
        losses = len(y_true_filtered) - wins

        profit = wins * 100 - losses * 110
        wagered = len(y_true_filtered) * 110

        return profit / wagered if wagered > 0 else 0.0

    def _compute_roi_kelly(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Compute ROI with Kelly criterion sizing (simplified)."""
        # Simplified Kelly: bet fraction = (bp - q) / b
        # where b = odds, p = predicted prob, q = 1-p
        # For -110 odds, b = 100/110 = 0.909

        b = 100 / 110
        total_profit = 0.0
        total_wagered = 0.0

        for prob, actual in zip(y_pred_proba, y_true):
            if prob >= 0.5:
                p = prob
                bet_home = True
            else:
                p = 1 - prob
                bet_home = False

            q = 1 - p
            kelly_fraction = max(0, (b * p - q) / b)

            if kelly_fraction > 0:
                # Cap at 5% of bankroll
                bet_size = min(kelly_fraction, 0.05) * 100

                won = (bet_home and actual == 1) or (not bet_home and actual == 0)

                if won:
                    total_profit += bet_size * b
                else:
                    total_profit -= bet_size

                total_wagered += bet_size

        return total_profit / total_wagered if total_wagered > 0 else 0.0

    def _compute_calibration_error(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Compute Expected Calibration Error (ECE)."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i+1])
            if mask.sum() > 0:
                bin_acc = y_true[mask].mean()
                bin_conf = y_pred_proba[mask].mean()
                bin_size = mask.sum() / len(y_true)
                ece += bin_size * abs(bin_acc - bin_conf)

        return ece

    def _compute_confidence_tier_accuracy(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        low: float,
        high: float
    ) -> float:
        """Compute accuracy for predictions in a confidence tier."""
        # Consider both sides of 0.5
        mask_high = (y_pred_proba >= low) & (y_pred_proba < high)
        mask_low = (y_pred_proba > (1 - high)) & (y_pred_proba <= (1 - low))
        mask = mask_high | mask_low

        if mask.sum() == 0:
            return 0.5  # No predictions in this tier

        y_pred = (y_pred_proba >= 0.5).astype(int)
        return (y_pred[mask] == y_true[mask]).mean()


def create_model_factory(model_type: str, config: Optional[dict] = None):
    """
    Create a model factory function for the specified model type.

    Args:
        model_type: Model type ('gbm', 'lr', 'ensemble', 'ft_transformer', 'tabnet')
        config: Optional model configuration

    Returns:
        Callable that creates a new model instance
    """
    def factory():
        if model_type in ['gbm', 'gradient_boosting', 'xgboost']:
            from models.architectures.gradient_boosting import GradientBoostingModel
            model_config = config or {}
            return GradientBoostingModel(**model_config)

        elif model_type in ['lr', 'logistic', 'logistic_regression']:
            from models.architectures.logistic_regression import LogisticRegressionModel
            model_config = config or {}
            return LogisticRegressionModel(**model_config)

        elif model_type in ['ensemble', 'stacking_ensemble', 'stacking']:
            from models.architectures.stacking_ensemble import StackingEnsemble
            model_config = config or {}
            # Default base models if not specified
            if 'base_models' not in model_config:
                from models.architectures.logistic_regression import LogisticRegressionModel
                from models.architectures.gradient_boosting import GradientBoostingModel
                model_config['base_models'] = {
                    'logistic': LogisticRegressionModel(),
                    'gbm': GradientBoostingModel(),
                }
            return StackingEnsemble(**model_config)

        elif model_type == 'ft_transformer':
            from models.architectures.ft_transformer import FTTransformerModel
            model_config = config or {'epochs': 50, 'patience': 10}
            return FTTransformerModel(**model_config)

        elif model_type == 'tabnet':
            from models.architectures.tabnet import TabNetModel
            model_config = config or {'epochs': 50, 'patience': 10}
            return TabNetModel(**model_config)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    return factory


def run_walk_forward_validation(
    features_path: str,
    model_type: str = 'gbm',
    output_dir: str = 'results/walk_forward/',
    min_train_seasons: int = 4,
    calibration_method: str = 'isotonic',
    start_test_season: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[WalkForwardResult]]:
    """
    Main entry point for walk-forward validation.

    Args:
        features_path: Path to features parquet file
        model_type: Model type to use
        output_dir: Output directory for results
        min_train_seasons: Minimum seasons for training
        calibration_method: Calibration method
        start_test_season: First test season (optional)

    Returns:
        Tuple of (summary_df, results_list)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Load features
    logger.info(f"Loading features from {features_path}")
    features_df = pd.read_parquet(features_path)
    logger.info(f"Loaded {len(features_df)} games")
    logger.info(f"Seasons: {sorted(features_df['season'].unique())}")

    # Create model factory
    model_factory = create_model_factory(model_type)

    # Run walk-forward validation
    validator = WalkForwardValidator(
        features_df=features_df,
        model_factory=model_factory,
        min_train_seasons=min_train_seasons,
        calibration_method=calibration_method,
    )

    results = validator.run_all_splits(start_test_season=start_test_season)
    summary = validator.summarize_results()

    # Save results
    summary.to_csv(f"{output_dir}/walk_forward_summary.csv", index=False)

    # Save all predictions
    all_predictions = pd.concat([r.predictions for r in results])
    all_predictions.to_parquet(f"{output_dir}/walk_forward_predictions.parquet")

    # Save config
    config = {
        'features_path': features_path,
        'model_type': model_type,
        'min_train_seasons': min_train_seasons,
        'calibration_method': calibration_method,
        'start_test_season': start_test_season,
        'n_splits': len(results),
        'total_test_games': sum(r.n_test_games for r in results),
        'mean_accuracy': summary['accuracy'].mean(),
        'mean_brier': summary['brier'].mean(),
        'timestamp': datetime.now().isoformat(),
    }
    with open(f"{output_dir}/walk_forward_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"\nResults saved to {output_dir}")

    return summary, results


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run walk-forward validation")
    parser.add_argument('--features', required=True, help='Path to features parquet')
    parser.add_argument('--model', default='gbm',
                       choices=['gbm', 'lr', 'ensemble', 'ft_transformer', 'tabnet'],
                       help='Model type')
    parser.add_argument('--output', default='results/walk_forward/', help='Output directory')
    parser.add_argument('--min-train-seasons', type=int, default=4, help='Min training seasons')
    parser.add_argument('--calibration', default='isotonic',
                       choices=['platt', 'isotonic', 'temperature'],
                       help='Calibration method')
    parser.add_argument('--start-test-season', type=int, default=None,
                       help='First test season')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    run_walk_forward_validation(
        features_path=args.features,
        model_type=args.model,
        output_dir=args.output,
        min_train_seasons=args.min_train_seasons,
        calibration_method=args.calibration,
        start_test_season=args.start_test_season,
    )


if __name__ == "__main__":
    main()
