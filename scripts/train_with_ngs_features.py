"""
Optimal Training Strategy for NGS Features

This script implements the optimal training approach for the new NGS/FTN/PFR features:

1. Feature Selection - Remove highly correlated features, select top predictors
2. Regularization - Stronger regularization to prevent overfitting with 400+ features
3. Walk-Forward Validation - Robust performance estimation
4. Calibration - Fix probability estimates
5. Ensemble - Combine baseline + NGS models

Key challenges:
- 402 new NGS features (total 467 numeric features)
- Many features are highly correlated (L3, L5, L8 windows)
- NGS data only available from 2016+
- Initial test showed +2.1% accuracy but worse Brier (overfitting signal)

Usage:
    # Full training with feature selection
    python scripts/train_with_ngs_features.py --mode full

    # Quick test
    python scripts/train_with_ngs_features.py --mode quick

    # Walk-forward validation
    python scripts/train_with_ngs_features.py --mode walk-forward
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Feature selection strategies for high-dimensional NGS features.

    Strategies:
    1. Correlation-based: Remove features correlated > threshold
    2. Importance-based: Keep top K features by model importance
    3. Group-based: Select top features per group (QB, RB, WR, baseline)
    """

    def __init__(self, correlation_threshold: float = 0.85):
        self.correlation_threshold = correlation_threshold
        self.selected_features = None
        self.removed_features = []

    def remove_correlated_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> List[str]:
        """
        Remove highly correlated features.

        Args:
            df: DataFrame with features
            feature_cols: List of feature columns

        Returns:
            List of selected feature columns
        """
        X = df[feature_cols].fillna(0)

        # Compute correlation matrix
        corr_matrix = X.corr().abs()

        # Get upper triangle
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features to drop
        to_drop = [col for col in upper.columns if any(upper[col] > self.correlation_threshold)]

        selected = [c for c in feature_cols if c not in to_drop]
        self.removed_features = to_drop

        logger.info(f"Removed {len(to_drop)} correlated features (threshold={self.correlation_threshold})")
        logger.info(f"Remaining features: {len(selected)}")

        return selected

    def select_by_importance(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        top_k: int = 100
    ) -> Tuple[List[str], np.ndarray]:
        """
        Select top K features by model importance.

        Uses a quick GBM to rank features by importance.

        Args:
            X_train: Training features
            y_train: Training target
            top_k: Number of features to keep

        Returns:
            Tuple of (selected_features, importance_scores)
        """
        logger.info(f"Selecting top {top_k} features by importance...")

        # Quick GBM for feature importance
        gbm = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=42
        )
        gbm.fit(X_train.fillna(0), y_train)

        # Get importance
        importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': gbm.feature_importances_
        }).sort_values('importance', ascending=False)

        # Select top K
        selected = importance.head(top_k)['feature'].tolist()

        logger.info(f"Top 10 features:")
        for i, row in importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        self.selected_features = selected
        return selected, gbm.feature_importances_

    def select_by_group(
        self,
        feature_cols: List[str],
        top_per_group: int = 30
    ) -> List[str]:
        """
        Select top features per group (baseline, QB, RB, WR).

        Ensures diversity across feature types.

        Args:
            feature_cols: All feature columns
            top_per_group: Features to keep per group

        Returns:
            List of selected features
        """
        groups = {
            'baseline': [c for c in feature_cols if 'ngs' not in c.lower() and 'ftn' not in c.lower() and 'pfr' not in c.lower()],
            'qb_ngs': [c for c in feature_cols if 'qb_ngs' in c.lower()],
            'rb_ngs': [c for c in feature_cols if 'rb_ngs' in c.lower()],
            'wr_ngs': [c for c in feature_cols if 'wr_ngs' in c.lower()],
            'ftn': [c for c in feature_cols if 'ftn' in c.lower()],
            'pfr': [c for c in feature_cols if 'pfr' in c.lower()],
        }

        selected = []
        for name, cols in groups.items():
            n_select = min(len(cols), top_per_group)
            selected.extend(cols[:n_select])
            logger.info(f"  {name}: {len(cols)} available, selected {n_select}")

        return selected


class OptimalNGSTrainer:
    """
    Optimal training pipeline for NGS features.

    Implements:
    1. Smart feature selection (correlation + importance)
    2. Regularized models
    3. Walk-forward validation
    4. Calibration
    5. Ensemble of baseline + NGS models
    """

    def __init__(
        self,
        features_path: str = "data/nfl/processed/game_features_with_ngs.parquet",
        output_dir: str = "models/artifacts/nfl_ngs_optimal"
    ):
        self.features_path = Path(features_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.df = None
        self.feature_cols = None
        self.selected_features = None

    def load_data(self) -> pd.DataFrame:
        """Load and prepare data."""
        logger.info(f"Loading data from {self.features_path}")
        self.df = pd.read_parquet(self.features_path)

        # Create target
        self.df['home_win'] = (self.df['home_score'] > self.df['away_score']).astype(int)

        # Identify feature columns
        exclude = ['game_id', 'season', 'week', 'date', 'gameday', 'home_team', 'away_team',
                   'home_score', 'away_score', 'home_win', 'game_type', 'close_spread', 'close_total']

        self.feature_cols = [c for c in self.df.columns
                            if c not in exclude
                            and self.df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

        logger.info(f"Games: {len(self.df)}")
        logger.info(f"Seasons: {sorted(self.df['season'].unique())}")
        logger.info(f"Total features: {len(self.feature_cols)}")

        # Count by type
        ngs_cols = [c for c in self.feature_cols if 'ngs' in c.lower()]
        ftn_cols = [c for c in self.feature_cols if 'ftn' in c.lower()]
        pfr_cols = [c for c in self.feature_cols if 'pfr' in c.lower()]
        baseline_cols = [c for c in self.feature_cols if c not in ngs_cols + ftn_cols + pfr_cols]

        logger.info(f"  Baseline: {len(baseline_cols)}")
        logger.info(f"  NGS: {len(ngs_cols)}")
        logger.info(f"  FTN: {len(ftn_cols)}")
        logger.info(f"  PFR: {len(pfr_cols)}")

        return self.df

    def prepare_splits(
        self,
        train_seasons: List[int],
        val_season: int,
        test_season: int
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Prepare train/val/test splits."""
        train_mask = self.df['season'].isin(train_seasons)
        val_mask = self.df['season'] == val_season
        test_mask = self.df['season'] == test_season

        X_train = self.df.loc[train_mask, self.feature_cols].fillna(0)
        y_train = self.df.loc[train_mask, 'home_win']
        X_val = self.df.loc[val_mask, self.feature_cols].fillna(0)
        y_val = self.df.loc[val_mask, 'home_win']
        X_test = self.df.loc[test_mask, self.feature_cols].fillna(0)
        y_test = self.df.loc[test_mask, 'home_win']

        logger.info(f"Train: {len(X_train)} ({train_seasons})")
        logger.info(f"Val: {len(X_val)} ({val_season})")
        logger.info(f"Test: {len(X_test)} ({test_season})")

        return X_train, y_train, X_val, y_val, X_test, y_test

    def select_features(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        strategy: str = 'combined',
        top_k: int = 100
    ) -> List[str]:
        """
        Select optimal features.

        Args:
            X_train: Training features
            y_train: Training target
            strategy: 'correlation', 'importance', 'combined', or 'all'
            top_k: Number of features to select

        Returns:
            List of selected feature names
        """
        selector = FeatureSelector(correlation_threshold=0.85)

        if strategy == 'all':
            return list(X_train.columns)

        if strategy == 'correlation':
            return selector.remove_correlated_features(X_train, list(X_train.columns))

        if strategy == 'importance':
            selected, _ = selector.select_by_importance(X_train, y_train, top_k=top_k)
            return selected

        if strategy == 'combined':
            # First remove correlated features
            uncorrelated = selector.remove_correlated_features(X_train, list(X_train.columns))

            # Then select by importance
            X_filtered = X_train[uncorrelated]
            selected, _ = selector.select_by_importance(X_filtered, y_train, top_k=top_k)
            return selected

        raise ValueError(f"Unknown strategy: {strategy}")

    def train_regularized_gbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> GradientBoostingClassifier:
        """
        Train GBM with strong regularization for high-dimensional features.
        """
        logger.info("Training regularized GBM...")

        # Regularization: lower depth, more estimators, lower learning rate
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,  # Shallow trees
            min_samples_split=20,  # Require more samples per split
            min_samples_leaf=10,   # Require more samples per leaf
            max_features='sqrt',   # Only use sqrt(n_features) per split
            subsample=0.8,         # Use 80% of samples per tree
            learning_rate=0.05,    # Low learning rate
            random_state=42
        )

        model.fit(X_train, y_train)

        # Evaluate
        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val))
        val_brier = brier_score_loss(y_val, model.predict_proba(X_val)[:, 1])

        logger.info(f"  Train accuracy: {train_acc:.3f}")
        logger.info(f"  Val accuracy: {val_acc:.3f}")
        logger.info(f"  Val Brier: {val_brier:.4f}")

        return model

    def train_regularized_lr(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Tuple[LogisticRegression, StandardScaler]:
        """
        Train L1-regularized logistic regression.
        L1 performs automatic feature selection.
        """
        logger.info("Training L1-regularized Logistic Regression...")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # L1 regularization (Lasso) for feature selection
        model = LogisticRegression(
            penalty='l1',
            solver='saga',
            C=0.1,  # Strong regularization
            max_iter=1000,
            random_state=42
        )

        model.fit(X_train_scaled, y_train)

        # Count non-zero coefficients
        n_nonzero = np.sum(model.coef_ != 0)
        logger.info(f"  Non-zero coefficients: {n_nonzero}/{len(model.coef_.flatten())}")

        # Evaluate
        train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
        val_acc = accuracy_score(y_val, model.predict(X_val_scaled))
        val_brier = brier_score_loss(y_val, model.predict_proba(X_val_scaled)[:, 1])

        logger.info(f"  Train accuracy: {train_acc:.3f}")
        logger.info(f"  Val accuracy: {val_acc:.3f}")
        logger.info(f"  Val Brier: {val_brier:.4f}")

        return model, scaler

    def apply_calibration(
        self,
        y_val: np.ndarray,
        y_pred_proba: np.ndarray,
        method: str = 'isotonic'
    ):
        """
        Apply probability calibration.

        Returns a calibrator that can transform probabilities.
        """
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression as LR

        if method == 'isotonic':
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(y_pred_proba, y_val)
        elif method == 'platt':
            calibrator = LR()
            calibrator.fit(y_pred_proba.reshape(-1, 1), y_val)
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        return calibrator

    def run_full_training(
        self,
        train_seasons: List[int] = list(range(2020, 2024)),
        val_season: int = 2024,
        test_season: int = 2024,
        feature_selection: str = 'combined',
        top_k_features: int = 100
    ) -> Dict:
        """
        Run complete optimal training pipeline.

        Args:
            train_seasons: Seasons to use for training
            val_season: Season for validation/calibration
            test_season: Season for final testing
            feature_selection: Feature selection strategy
            top_k_features: Number of features to select

        Returns:
            Dict with results and trained models
        """
        logger.info("\n" + "="*60)
        logger.info("OPTIMAL NGS TRAINING PIPELINE")
        logger.info("="*60)

        # Load data
        self.load_data()

        # Prepare splits
        X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_splits(
            train_seasons, val_season, test_season
        )

        # Feature selection
        logger.info(f"\n--- Feature Selection ({feature_selection}) ---")
        selected_features = self.select_features(X_train, y_train, feature_selection, top_k_features)
        self.selected_features = selected_features

        logger.info(f"Selected {len(selected_features)} features")

        # Filter to selected features
        X_train_sel = X_train[selected_features]
        X_val_sel = X_val[selected_features]
        X_test_sel = X_test[selected_features]

        # Train models
        logger.info("\n--- Training Models ---")

        # 1. Regularized GBM
        gbm = self.train_regularized_gbm(X_train_sel, y_train, X_val_sel, y_val)

        # 2. Regularized LR
        lr, scaler = self.train_regularized_lr(X_train_sel, y_train, X_val_sel, y_val)

        # 3. Ensemble (average)
        logger.info("\nTraining Ensemble (GBM + LR average)...")

        gbm_prob = gbm.predict_proba(X_val_sel)[:, 1]
        lr_prob = lr.predict_proba(scaler.transform(X_val_sel))[:, 1]
        ensemble_prob_val = 0.6 * gbm_prob + 0.4 * lr_prob

        # Calibrate
        logger.info("\n--- Calibration ---")
        calibrator = self.apply_calibration(y_val.values, ensemble_prob_val, method='isotonic')

        # Evaluate on test (if different from val)
        logger.info("\n--- Test Set Evaluation ---")

        gbm_prob_test = gbm.predict_proba(X_test_sel)[:, 1]
        lr_prob_test = lr.predict_proba(scaler.transform(X_test_sel))[:, 1]
        ensemble_prob_test = 0.6 * gbm_prob_test + 0.4 * lr_prob_test

        # Apply calibration
        calibrated_prob = calibrator.predict(ensemble_prob_test)
        calibrated_pred = (calibrated_prob >= 0.5).astype(int)

        # Metrics
        accuracy = accuracy_score(y_test, calibrated_pred)
        brier = brier_score_loss(y_test, calibrated_prob)
        ll = log_loss(y_test, np.clip(calibrated_prob, 1e-7, 1-1e-7))

        logger.info(f"  Accuracy: {accuracy:.1%} ({sum(calibrated_pred == y_test)}/{len(y_test)})")
        logger.info(f"  Brier Score: {brier:.4f}")
        logger.info(f"  Log Loss: {ll:.4f}")

        # Compare to uncalibrated
        uncal_brier = brier_score_loss(y_test, ensemble_prob_test)
        logger.info(f"  Brier improvement from calibration: {uncal_brier - brier:.4f}")

        # Save results
        results = {
            'accuracy': accuracy,
            'brier': brier,
            'log_loss': ll,
            'n_features': len(selected_features),
            'selected_features': selected_features,
            'train_seasons': train_seasons,
            'test_season': test_season,
        }

        # Save models
        import pickle
        with open(self.output_dir / "gbm.pkl", 'wb') as f:
            pickle.dump(gbm, f)
        with open(self.output_dir / "lr.pkl", 'wb') as f:
            pickle.dump((lr, scaler), f)
        with open(self.output_dir / "calibrator.pkl", 'wb') as f:
            pickle.dump(calibrator, f)
        with open(self.output_dir / "selected_features.pkl", 'wb') as f:
            pickle.dump(selected_features, f)

        logger.info(f"\nModels saved to {self.output_dir}")

        return results

    def run_walk_forward(
        self,
        min_train_seasons: int = 4,
        start_test_season: int = 2021
    ) -> pd.DataFrame:
        """
        Run walk-forward validation.

        Args:
            min_train_seasons: Minimum seasons for training
            start_test_season: First test season

        Returns:
            DataFrame with results per split
        """
        logger.info("\n" + "="*60)
        logger.info("WALK-FORWARD VALIDATION WITH NGS FEATURES")
        logger.info("="*60)

        self.load_data()

        # Filter to seasons with NGS data
        df = self.df[self.df['season'] >= 2020].copy()
        seasons = sorted(df['season'].unique())

        results = []

        for test_season in seasons:
            if test_season < start_test_season:
                continue

            train_seasons = [s for s in seasons if s < test_season]
            if len(train_seasons) < min_train_seasons:
                continue

            logger.info(f"\n--- Split: Train {train_seasons}, Test {test_season} ---")

            # Use last training season for validation
            val_season = max(train_seasons)
            train_train_seasons = [s for s in train_seasons if s < val_season]

            if len(train_train_seasons) < 2:
                continue

            # Prepare data
            train_mask = df['season'].isin(train_train_seasons)
            val_mask = df['season'] == val_season
            test_mask = df['season'] == test_season

            X_train = df.loc[train_mask, self.feature_cols].fillna(0)
            y_train = df.loc[train_mask, 'home_win']
            X_val = df.loc[val_mask, self.feature_cols].fillna(0)
            y_val = df.loc[val_mask, 'home_win']
            X_test = df.loc[test_mask, self.feature_cols].fillna(0)
            y_test = df.loc[test_mask, 'home_win']

            # Feature selection on this split's training data
            selected = self.select_features(X_train, y_train, strategy='combined', top_k=80)

            X_train_sel = X_train[selected]
            X_val_sel = X_val[selected]
            X_test_sel = X_test[selected]

            # Train GBM
            gbm = self.train_regularized_gbm(X_train_sel, y_train, X_val_sel, y_val)

            # Predict and calibrate
            val_prob = gbm.predict_proba(X_val_sel)[:, 1]
            test_prob = gbm.predict_proba(X_test_sel)[:, 1]

            calibrator = self.apply_calibration(y_val.values, val_prob)
            calibrated_prob = calibrator.predict(test_prob)
            calibrated_pred = (calibrated_prob >= 0.5).astype(int)

            acc = accuracy_score(y_test, calibrated_pred)
            brier = brier_score_loss(y_test, calibrated_prob)

            results.append({
                'train_seasons': str(train_train_seasons),
                'val_season': val_season,
                'test_season': test_season,
                'n_train': len(X_train),
                'n_test': len(X_test),
                'n_features': len(selected),
                'accuracy': acc,
                'brier': brier,
            })

            logger.info(f"  Test Accuracy: {acc:.1%}, Brier: {brier:.4f}")

        results_df = pd.DataFrame(results)

        logger.info("\n" + "="*60)
        logger.info("WALK-FORWARD SUMMARY")
        logger.info("="*60)
        logger.info(f"Splits: {len(results_df)}")
        logger.info(f"Mean Accuracy: {results_df['accuracy'].mean():.1%} (std: {results_df['accuracy'].std():.1%})")
        logger.info(f"Mean Brier: {results_df['brier'].mean():.4f}")

        results_df.to_csv(self.output_dir / "walk_forward_results.csv", index=False)

        return results_df


def main():
    parser = argparse.ArgumentParser(description="Optimal NGS Feature Training")
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'quick', 'walk-forward'],
                        help='Training mode')
    parser.add_argument('--features', type=str,
                        default='data/nfl/processed/game_features_with_ngs.parquet',
                        help='Path to features file')
    parser.add_argument('--output', type=str,
                        default='models/artifacts/nfl_ngs_optimal',
                        help='Output directory')
    parser.add_argument('--top-k', type=int, default=100,
                        help='Number of features to select')

    args = parser.parse_args()

    trainer = OptimalNGSTrainer(
        features_path=args.features,
        output_dir=args.output
    )

    if args.mode == 'full':
        results = trainer.run_full_training(
            train_seasons=list(range(2020, 2024)),
            val_season=2024,
            test_season=2024,
            feature_selection='combined',
            top_k_features=args.top_k
        )
        print(f"\n=== Final Results ===")
        print(f"Accuracy: {results['accuracy']:.1%}")
        print(f"Brier Score: {results['brier']:.4f}")
        print(f"Features used: {results['n_features']}")

    elif args.mode == 'quick':
        # Quick test with fewer features
        results = trainer.run_full_training(
            train_seasons=[2022, 2023],
            val_season=2024,
            test_season=2024,
            feature_selection='importance',
            top_k_features=50
        )

    elif args.mode == 'walk-forward':
        results_df = trainer.run_walk_forward(
            min_train_seasons=2,
            start_test_season=2022
        )
        print(f"\n=== Walk-Forward Results ===")
        print(results_df.to_string())


if __name__ == "__main__":
    main()
