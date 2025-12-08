"""
Production NGS Model

L1-regularized logistic regression model trained on NGS features.
This model is optimized for production use with automatic feature selection.

Key characteristics:
- Uses L1 regularization (C=0.05) for automatic feature selection
- Selects ~20-30 features from 460+ available
- Includes isotonic calibration for well-calibrated probabilities
- Trained on all available data (2020-2024)

Usage:
    from models.ngs_production_model import NGSProductionModel

    # Load trained model
    model = NGSProductionModel.load("models/artifacts/nfl_ngs_production")

    # Predict
    proba = model.predict_proba(game_features)
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata about the trained model."""
    train_seasons: List[int]
    val_season: int
    n_train_games: int
    n_val_games: int
    n_features_input: int
    n_features_selected: int
    selected_features: List[str]
    val_accuracy: float
    val_brier: float
    regularization_c: float
    trained_at: str


class NGSProductionModel:
    """
    Production model using L1-regularized logistic regression with NGS features.

    This model:
    1. Uses strong L1 regularization for automatic feature selection
    2. Applies isotonic calibration for well-calibrated probabilities
    3. Includes all preprocessing (scaling, imputation) in the pipeline
    """

    def __init__(
        self,
        regularization_c: float = 0.05,
        max_iter: int = 2000,
        random_state: int = 42
    ):
        """
        Initialize the production model.

        Args:
            regularization_c: Regularization strength (lower = stronger regularization)
            max_iter: Maximum iterations for convergence
            random_state: Random seed for reproducibility
        """
        self.regularization_c = regularization_c
        self.max_iter = max_iter
        self.random_state = random_state

        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.calibrator: Optional[IsotonicRegression] = None
        self.feature_cols: Optional[List[str]] = None
        self.selected_features: Optional[List[str]] = None
        self.metadata: Optional[ModelMetadata] = None

        self._is_fitted = False

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get numeric feature columns, excluding metadata and target."""
        exclude = [
            'game_id', 'season', 'week', 'date', 'gameday',
            'home_team', 'away_team', 'home_score', 'away_score',
            'home_win', 'game_type', 'close_spread', 'close_total',
            'open_spread', 'open_total'
        ]

        feature_cols = [
            c for c in df.columns
            if c not in exclude
            and df[c].dtype in ['float64', 'int64', 'float32', 'int32']
        ]

        return feature_cols

    def fit(
        self,
        features_df: pd.DataFrame,
        train_seasons: List[int],
        val_season: int
    ) -> 'NGSProductionModel':
        """
        Train the production model.

        Args:
            features_df: DataFrame with all features and game data
            train_seasons: List of seasons to use for training
            val_season: Season to use for validation/calibration

        Returns:
            self
        """
        logger.info("=" * 60)
        logger.info("Training NGS Production Model")
        logger.info("=" * 60)

        # Create target
        if 'home_win' not in features_df.columns:
            features_df = features_df.copy()
            features_df['home_win'] = (features_df['home_score'] > features_df['away_score']).astype(int)

        # Get feature columns
        self.feature_cols = self._get_feature_columns(features_df)
        logger.info(f"Input features: {len(self.feature_cols)}")

        # Split data
        train_mask = features_df['season'].isin(train_seasons)
        val_mask = features_df['season'] == val_season

        X_train = features_df.loc[train_mask, self.feature_cols].copy()
        y_train = features_df.loc[train_mask, 'home_win']
        X_val = features_df.loc[val_mask, self.feature_cols].copy()
        y_val = features_df.loc[val_mask, 'home_win']

        logger.info(f"Train: {len(X_train)} games ({train_seasons})")
        logger.info(f"Val: {len(X_val)} games ({val_season})")

        # Handle missing values
        X_train = X_train.fillna(0)
        X_val = X_val.fillna(0)

        # Scale features
        logger.info("Scaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Train L1-regularized model
        logger.info(f"Training L1 Logistic Regression (C={self.regularization_c})...")
        self.model = LogisticRegression(
            penalty='l1',
            solver='saga',
            C=self.regularization_c,
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)

        # Get selected features (non-zero coefficients)
        nonzero_mask = self.model.coef_.flatten() != 0
        self.selected_features = [
            self.feature_cols[i]
            for i, is_nonzero in enumerate(nonzero_mask)
            if is_nonzero
        ]
        logger.info(f"Selected {len(self.selected_features)} features")

        # Show top features
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'coef': np.abs(self.model.coef_.flatten())
        }).sort_values('coef', ascending=False)

        logger.info("Top 10 features:")
        for _, row in importance.head(10).iterrows():
            if row['coef'] > 0:
                logger.info(f"  {row['feature']}: {row['coef']:.4f}")

        # Calibrate on validation set
        logger.info("Calibrating probabilities...")
        val_proba = self.model.predict_proba(X_val_scaled)[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(val_proba, y_val)

        # Evaluate
        cal_proba = self.calibrator.predict(val_proba)
        cal_pred = (cal_proba >= 0.5).astype(int)

        val_accuracy = accuracy_score(y_val, cal_pred)
        val_brier = brier_score_loss(y_val, cal_proba)

        logger.info(f"\nValidation Results:")
        logger.info(f"  Accuracy: {val_accuracy:.1%} ({sum(cal_pred == y_val)}/{len(y_val)})")
        logger.info(f"  Brier Score: {val_brier:.4f}")

        # Store metadata
        from datetime import datetime
        self.metadata = ModelMetadata(
            train_seasons=train_seasons,
            val_season=val_season,
            n_train_games=len(X_train),
            n_val_games=len(X_val),
            n_features_input=len(self.feature_cols),
            n_features_selected=len(self.selected_features),
            selected_features=self.selected_features,
            val_accuracy=val_accuracy,
            val_brier=val_brier,
            regularization_c=self.regularization_c,
            trained_at=datetime.now().isoformat()
        )

        self._is_fitted = True
        logger.info("\nModel training complete!")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict calibrated probabilities for home team win.

        Args:
            X: DataFrame with features (must have same columns as training data)

        Returns:
            Array of probabilities for home team win
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first or load a trained model.")

        # Ensure we have the right columns
        missing_cols = set(self.feature_cols) - set(X.columns)
        if missing_cols:
            logger.warning(f"Missing {len(missing_cols)} features, filling with 0")
            for col in missing_cols:
                X = X.copy()
                X[col] = 0

        # Select and order columns
        X_features = X[self.feature_cols].fillna(0)

        # Scale
        X_scaled = self.scaler.transform(X_features)

        # Predict
        raw_proba = self.model.predict_proba(X_scaled)[:, 1]

        # Calibrate
        cal_proba = self.calibrator.predict(raw_proba)

        return cal_proba

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict home team win (1) or loss (0).

        Args:
            X: DataFrame with features
            threshold: Probability threshold for prediction

        Returns:
            Array of predictions (1 = home win, 0 = away win)
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def predict_game(
        self,
        game_features: pd.DataFrame,
        home_team: str,
        away_team: str
    ) -> Dict[str, Any]:
        """
        Predict a single game with detailed output.

        Args:
            game_features: DataFrame with one row of features
            home_team: Home team name
            away_team: Away team name

        Returns:
            Dict with prediction details
        """
        proba = self.predict_proba(game_features)[0]

        # Determine prediction
        if proba >= 0.5:
            predicted_winner = home_team
            confidence = proba
        else:
            predicted_winner = away_team
            confidence = 1 - proba

        # Calculate edge (for betting)
        edge = abs(proba - 0.5)

        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_win_prob': proba,
            'away_win_prob': 1 - proba,
            'predicted_winner': predicted_winner,
            'confidence': confidence,
            'edge': edge,
            'bet_recommendation': 'BET' if edge >= 0.05 else 'PASS'
        }

    def save(self, path: str) -> None:
        """
        Save the model to disk.

        Args:
            path: Directory path to save model artifacts
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model components
        with open(path / "model.pkl", 'wb') as f:
            pickle.dump(self.model, f)

        with open(path / "scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)

        with open(path / "calibrator.pkl", 'wb') as f:
            pickle.dump(self.calibrator, f)

        with open(path / "feature_cols.pkl", 'wb') as f:
            pickle.dump(self.feature_cols, f)

        with open(path / "selected_features.pkl", 'wb') as f:
            pickle.dump(self.selected_features, f)

        # Save metadata as JSON
        if self.metadata:
            metadata_dict = {
                'train_seasons': self.metadata.train_seasons,
                'val_season': self.metadata.val_season,
                'n_train_games': self.metadata.n_train_games,
                'n_val_games': self.metadata.n_val_games,
                'n_features_input': self.metadata.n_features_input,
                'n_features_selected': self.metadata.n_features_selected,
                'selected_features': self.metadata.selected_features,
                'val_accuracy': self.metadata.val_accuracy,
                'val_brier': self.metadata.val_brier,
                'regularization_c': self.metadata.regularization_c,
                'trained_at': self.metadata.trained_at
            }
            with open(path / "metadata.json", 'w') as f:
                json.dump(metadata_dict, f, indent=2)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'NGSProductionModel':
        """
        Load a trained model from disk.

        Args:
            path: Directory path containing model artifacts

        Returns:
            Loaded NGSProductionModel instance
        """
        path = Path(path)

        instance = cls()

        with open(path / "model.pkl", 'rb') as f:
            instance.model = pickle.load(f)

        with open(path / "scaler.pkl", 'rb') as f:
            instance.scaler = pickle.load(f)

        with open(path / "calibrator.pkl", 'rb') as f:
            instance.calibrator = pickle.load(f)

        with open(path / "feature_cols.pkl", 'rb') as f:
            instance.feature_cols = pickle.load(f)

        with open(path / "selected_features.pkl", 'rb') as f:
            instance.selected_features = pickle.load(f)

        # Load metadata
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            instance.metadata = ModelMetadata(**metadata_dict)

        instance._is_fitted = True

        logger.info(f"Model loaded from {path}")
        logger.info(f"  Selected features: {len(instance.selected_features)}")
        if instance.metadata:
            logger.info(f"  Val accuracy: {instance.metadata.val_accuracy:.1%}")

        return instance

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance (absolute coefficient values).

        Returns:
            DataFrame with feature names and importance scores
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted.")

        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'coefficient': self.model.coef_.flatten(),
            'abs_importance': np.abs(self.model.coef_.flatten())
        }).sort_values('abs_importance', ascending=False)

        importance['selected'] = importance['abs_importance'] > 0

        return importance


def train_production_model(
    features_path: str = "data/nfl/processed/game_features_with_ngs.parquet",
    output_dir: str = "models/artifacts/nfl_ngs_production",
    train_seasons: Optional[List[int]] = None,
    val_season: int = 2024
) -> NGSProductionModel:
    """
    Train and save the production model.

    Args:
        features_path: Path to features parquet file
        output_dir: Directory to save model
        train_seasons: Training seasons (default: 2020-2023)
        val_season: Validation season

    Returns:
        Trained model
    """
    if train_seasons is None:
        train_seasons = [2020, 2021, 2022, 2023]

    # Load data
    logger.info(f"Loading features from {features_path}")
    df = pd.read_parquet(features_path)

    # Filter to seasons with NGS data
    df = df[df['season'] >= 2020]
    logger.info(f"Games with NGS data: {len(df)}")

    # Train model
    model = NGSProductionModel(regularization_c=0.05)
    model.fit(df, train_seasons=train_seasons, val_season=val_season)

    # Save
    model.save(output_dir)

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train NGS Production Model")
    parser.add_argument('--features', default='data/nfl/processed/game_features_with_ngs.parquet')
    parser.add_argument('--output', default='models/artifacts/nfl_ngs_production')
    parser.add_argument('--val-season', type=int, default=2024)

    args = parser.parse_args()

    train_production_model(
        features_path=args.features,
        output_dir=args.output,
        val_season=args.val_season
    )
