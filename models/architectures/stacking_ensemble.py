"""
Stacking Ensemble with Meta-Model

Advanced ensemble that combines multiple base models using a meta-learner.
Supports:
- Any number of base models following the BaseModel interface
- Logistic regression or MLP meta-models
- Optional inclusion of original features in stacking
- Probability outputs from all base models and the ensemble
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Optional, Union, Literal
from pathlib import Path
import pickle
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.base import BaseModel

logger = logging.getLogger(__name__)


class MLPMetaModel(nn.Module):
    """
    Small MLP for meta-learning over base model predictions.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [16, 8],
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class StackingEnsemble(BaseModel):
    """
    Stacking ensemble that combines multiple base models with a meta-learner.

    The ensemble:
    1. Collects predictions from all base models
    2. Optionally includes original features
    3. Trains a meta-model on the stacked predictions
    4. Returns calibrated ensemble probabilities

    Args:
        base_models: Dictionary mapping model names to trained BaseModel instances
        meta_model_type: Type of meta-learner ("logistic" or "mlp")
        include_features: Whether to include original features in stacking
        feature_fraction: Fraction of original features to include (if include_features=True)
        mlp_hidden_dims: Hidden layer dimensions for MLP meta-model
        mlp_dropout: Dropout rate for MLP
        mlp_epochs: Training epochs for MLP
        mlp_learning_rate: Learning rate for MLP
        random_state: Random seed for reproducibility
    """

    def __init__(
        self,
        base_models: Optional[Dict[str, BaseModel]] = None,
        meta_model_type: Literal["logistic", "mlp"] = "logistic",
        include_features: bool = False,
        feature_fraction: float = 0.0,
        mlp_hidden_dims: List[int] = [16, 8],
        mlp_dropout: float = 0.1,
        mlp_epochs: int = 50,
        mlp_learning_rate: float = 1e-3,
        random_state: int = 42,
    ):
        self.base_models = base_models or {}
        self.meta_model_type = meta_model_type
        self.include_features = include_features
        self.feature_fraction = feature_fraction
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_dropout = mlp_dropout
        self.mlp_epochs = mlp_epochs
        self.mlp_learning_rate = mlp_learning_rate
        self.random_state = random_state

        # Meta model (set during fit)
        self.meta_model = None
        self.scaler = StandardScaler()

        # Track which features to use if include_features is True
        self.selected_feature_indices: Optional[np.ndarray] = None
        self.n_base_models: int = 0
        self.model_names: List[str] = []

        # Set random seeds
        np.random.seed(random_state)
        torch.manual_seed(random_state)

    def add_model(self, name: str, model: BaseModel) -> None:
        """Add a base model to the ensemble."""
        self.base_models[name] = model

    def remove_model(self, name: str) -> None:
        """Remove a base model from the ensemble."""
        if name in self.base_models:
            del self.base_models[name]

    def _get_base_predictions(self, X) -> np.ndarray:
        """
        Get predictions from all base models.

        Returns:
            Array of shape (n_samples, n_base_models)
        """
        if len(self.base_models) == 0:
            raise ValueError("No base models in ensemble")

        predictions = []
        for name in self.model_names:
            model = self.base_models[name]
            pred = model.predict_proba(X)
            predictions.append(pred)

        return np.column_stack(predictions)

    def _prepare_meta_features(self, X, base_preds: np.ndarray) -> np.ndarray:
        """
        Prepare feature matrix for meta-model.

        Combines base model predictions with (optionally) original features.
        """
        if self.include_features and self.feature_fraction > 0:
            if hasattr(X, 'values'):
                X_arr = X.values
            else:
                X_arr = np.asarray(X)

            # Select subset of features
            if self.selected_feature_indices is not None:
                X_subset = X_arr[:, self.selected_feature_indices]
            else:
                X_subset = X_arr

            return np.hstack([base_preds, X_subset])
        else:
            return base_preds

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit the stacking ensemble.

        CRITICAL: X must be TRAINING data only. X_val is used ONLY for early stopping
        (MLP models) or evaluation, never for fitting the meta-model or scaler.

        Note: Base models should already be trained. This method only
        trains the meta-model on their predictions.

        Args:
            X: TRAINING feature matrix (used for base model predictions and meta-model training)
            y: TRAINING target labels
            X_val: Optional validation features (used ONLY for early stopping/evaluation, not training)
            y_val: Optional validation labels (used ONLY for early stopping/evaluation, not training)
        """
        if len(self.base_models) == 0:
            raise ValueError("No base models in ensemble. Add models with add_model()")

        # VALIDATION: Check for overlap between X and X_val
        # NOTE: Validation CAN be a subset of training (standard for calibration/tuning)
        # This is acceptable because:
        # - Meta-model trains ONLY on X (training data)
        # - X_val is used ONLY for early stopping/evaluation, never for training
        # - Scalers are fit ONLY on training data
        # We only warn if validation is a proper subset (expected) vs. unexpected partial overlap
        if X_val is not None and hasattr(X, 'index') and hasattr(X_val, 'index'):
            train_indices = set(X.index)
            val_indices = set(X_val.index)
            overlap = train_indices & val_indices
            
            if overlap:
                # Check if validation is a proper subset of training (acceptable)
                if val_indices.issubset(train_indices):
                    logger.info(f"  Note: Validation set ({len(val_indices)} samples) is a subset of training set ({len(train_indices)} samples)")
                    logger.info(f"  This is acceptable: validation used only for early stopping/evaluation, not training")
                else:
                    # Partial overlap might indicate a bug - warn but don't fail
                    logger.warning(f"  Warning: Partial overlap detected between training and validation sets")
                    logger.warning(f"  Overlap: {len(overlap)} samples. Ensure validation is only used for evaluation, not training.")

        # Store model order for consistent predictions
        self.model_names = sorted(self.base_models.keys())
        self.n_base_models = len(self.model_names)

        logger.info(f"Fitting stacking ensemble with {self.n_base_models} base models")
        logger.info(f"  Models: {self.model_names}")
        logger.info(f"  Meta-model: {self.meta_model_type}")
        logger.info(f"  Training on {len(X)} samples")

        # Get base model predictions on TRAINING data
        base_preds = self._get_base_predictions(X)
        logger.info(f"  Base predictions shape: {base_preds.shape}")

        # Select feature subset if needed
        if self.include_features and self.feature_fraction > 0:
            if hasattr(X, 'values'):
                n_features = X.shape[1]
            else:
                n_features = np.asarray(X).shape[1]

            n_select = max(1, int(n_features * self.feature_fraction))
            self.selected_feature_indices = np.random.choice(
                n_features, size=n_select, replace=False
            )
            logger.info(f"  Including {n_select} original features in stacking")

        # Prepare meta features from TRAINING data
        meta_features = self._prepare_meta_features(X, base_preds)
        y_arr = np.asarray(y)

        # CRITICAL: Fit scaler ONLY on training meta-features
        # This ensures scaler statistics come from training data only
        meta_features_scaled = self.scaler.fit_transform(meta_features)

        # Train meta-model on TRAINING data only
        if self.meta_model_type == "logistic":
            self._fit_logistic(meta_features_scaled, y_arr)
        else:
            # Prepare validation data if available (for early stopping only)
            # CRITICAL: Use transform() only, never fit_transform() on validation data
            if X_val is not None and y_val is not None:
                val_base_preds = self._get_base_predictions(X_val)
                val_meta_features = self._prepare_meta_features(X_val, val_base_preds)
                # Transform validation features using scaler fit on training data
                val_meta_features_scaled = self.scaler.transform(val_meta_features)
                y_val_arr = np.asarray(y_val)
            else:
                val_meta_features_scaled = None
                y_val_arr = None

            # MLP uses validation data for early stopping only, not for training
            self._fit_mlp(meta_features_scaled, y_arr,
                          val_meta_features_scaled, y_val_arr)

        logger.info("  Stacking ensemble training complete")
        return self

    def _fit_logistic(self, X: np.ndarray, y: np.ndarray):
        """Fit logistic regression meta-model."""
        self.meta_model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=self.random_state,
        )
        self.meta_model.fit(X, y)

        # Log coefficients for interpretability
        logger.info("  Meta-model coefficients:")
        for i, name in enumerate(self.model_names):
            logger.info(f"    {name}: {self.meta_model.coef_[0][i]:.4f}")

    def _fit_mlp(self, X: np.ndarray, y: np.ndarray,
                 X_val: Optional[np.ndarray], y_val: Optional[np.ndarray]):
        """Fit MLP meta-model."""
        input_dim = X.shape[1]
        self.meta_model = MLPMetaModel(
            input_dim=input_dim,
            hidden_dims=self.mlp_hidden_dims,
            dropout=self.mlp_dropout,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.meta_model.to(device)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(
            self.meta_model.parameters(),
            lr=self.mlp_learning_rate,
        )
        criterion = nn.BCELoss()

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

        best_val_loss = float('inf')
        best_state = None

        for epoch in range(self.mlp_epochs):
            self.meta_model.train()
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                preds = self.meta_model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

            if has_val:
                self.meta_model.eval()
                with torch.no_grad():
                    val_preds = self.meta_model(X_val_tensor)
                    val_loss = criterion(val_preds, y_val_tensor).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.meta_model.state_dict().items()}

        if best_state is not None:
            self.meta_model.load_state_dict(best_state)
            self.meta_model.to(device)

    def predict_proba(self, X) -> np.ndarray:
        """
        Predict ensemble probabilities.

        Args:
            X: Feature matrix

        Returns:
            Array of shape (n_samples,) with ensemble probabilities
        """
        if self.meta_model is None:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        # Get base predictions
        base_preds = self._get_base_predictions(X)

        # Prepare meta features
        meta_features = self._prepare_meta_features(X, base_preds)
        meta_features_scaled = self.scaler.transform(meta_features)

        # Get meta-model prediction
        if self.meta_model_type == "logistic":
            probs = self.meta_model.predict_proba(meta_features_scaled)[:, 1]
        else:
            device = next(self.meta_model.parameters()).device
            X_tensor = torch.tensor(meta_features_scaled, dtype=torch.float32).to(device)
            self.meta_model.eval()
            with torch.no_grad():
                probs = self.meta_model(X_tensor).cpu().numpy()

        return probs

    def predict_all(self, X) -> Dict[str, np.ndarray]:
        """
        Get predictions from all models (base + ensemble).

        Args:
            X: Feature matrix

        Returns:
            Dictionary with model name -> predictions
        """
        results = {}

        # Base model predictions
        for name in self.model_names:
            model = self.base_models[name]
            results[name] = model.predict_proba(X)

        # Ensemble prediction
        results['ensemble'] = self.predict_proba(X)

        return results

    def get_model_weights(self) -> Optional[Dict[str, float]]:
        """
        Get the learned weights for each base model.

        Only available for logistic meta-model.
        """
        if self.meta_model_type != "logistic" or self.meta_model is None:
            return None

        weights = self.meta_model.coef_[0][:self.n_base_models]

        # Normalize to sum to 1 (approximately)
        weights = np.abs(weights)
        weights = weights / weights.sum()

        return dict(zip(self.model_names, weights))

    def save(self, path: Path):
        """Save ensemble to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save base models separately
        base_model_paths = {}
        for name, model in self.base_models.items():
            model_path = path.parent / f"{path.stem}_base_{name}.pkl"
            model.save(model_path)
            base_model_paths[name] = str(model_path)

        save_dict = {
            'config': {
                'meta_model_type': self.meta_model_type,
                'include_features': self.include_features,
                'feature_fraction': self.feature_fraction,
                'mlp_hidden_dims': self.mlp_hidden_dims,
                'mlp_dropout': self.mlp_dropout,
                'mlp_epochs': self.mlp_epochs,
                'mlp_learning_rate': self.mlp_learning_rate,
                'random_state': self.random_state,
            },
            'base_model_paths': base_model_paths,
            'model_names': self.model_names,
            'n_base_models': self.n_base_models,
            'selected_feature_indices': self.selected_feature_indices,
            'scaler': self.scaler,
        }

        # Save meta-model
        if self.meta_model_type == "logistic":
            save_dict['meta_model'] = self.meta_model
        elif self.meta_model is not None:
            save_dict['meta_model_state'] = self.meta_model.state_dict()
            save_dict['meta_input_dim'] = list(self.meta_model.parameters())[0].shape[1]

        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

    @classmethod
    def load(cls, path: Path, base_model_loader=None) -> 'StackingEnsemble':
        """
        Load ensemble from disk.

        Args:
            path: Path to saved ensemble
            base_model_loader: Optional function to load base models
                               (default uses BaseModel.load)
        """
        path = Path(path)

        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        # Create instance
        instance = cls(**save_dict['config'])
        instance.model_names = save_dict['model_names']
        instance.n_base_models = save_dict['n_base_models']
        instance.selected_feature_indices = save_dict['selected_feature_indices']
        instance.scaler = save_dict['scaler']

        # Load base models
        loader = base_model_loader or BaseModel.load
        for name, model_path in save_dict['base_model_paths'].items():
            instance.base_models[name] = loader(model_path)

        # Load meta-model
        if save_dict['config']['meta_model_type'] == "logistic":
            instance.meta_model = save_dict['meta_model']
        elif 'meta_model_state' in save_dict:
            instance.meta_model = MLPMetaModel(
                input_dim=save_dict['meta_input_dim'],
                hidden_dims=save_dict['config']['mlp_hidden_dims'],
                dropout=save_dict['config']['mlp_dropout'],
            )
            instance.meta_model.load_state_dict(save_dict['meta_model_state'])

        return instance
