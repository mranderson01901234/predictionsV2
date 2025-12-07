"""
TabNet Model for Tabular Data

TabNet (Arik & Pfister, 2019) uses sequential attention to select features
at each decision step, providing interpretability while achieving strong
performance on tabular data.

This implementation uses pytorch-tabnet library when available,
with a fallback to a simplified native implementation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, List, Dict, Any
from pathlib import Path
import pickle
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.base import BaseModel

logger = logging.getLogger(__name__)


# Try to import pytorch-tabnet
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    logger.warning("pytorch-tabnet not installed. Using simplified implementation.")


class TabNetModel(BaseModel):
    """
    TabNet model for NFL game prediction.

    Uses pytorch-tabnet library when available, otherwise falls back
    to a simplified native PyTorch implementation.

    Args:
        n_d: Width of decision prediction layer (default: 8)
        n_a: Width of attention embedding (default: 8)
        n_steps: Number of decision steps (default: 3)
        gamma: Coefficient for feature reusage in mask (default: 1.3)
        n_independent: Number of independent GLU layers (default: 1)
        n_shared: Number of shared GLU layers (default: 1)
        lambda_sparse: Sparsity regularization (default: 1e-3)
        momentum: Batch normalization momentum (default: 0.02)
        learning_rate: Learning rate (default: 2e-2)
        batch_size: Training batch size (default: 256)
        epochs: Number of training epochs (default: 100)
        patience: Early stopping patience (default: 15)
        random_state: Random seed for reproducibility (default: 42)
        device: Device to train on ("cuda", "cpu", or "auto")
    """

    def __init__(
        self,
        n_d: int = 8,
        n_a: int = 8,
        n_steps: int = 3,
        gamma: float = 1.3,
        n_independent: int = 1,
        n_shared: int = 1,
        lambda_sparse: float = 1e-3,
        momentum: float = 0.02,
        learning_rate: float = 2e-2,
        batch_size: int = 256,
        epochs: int = 100,
        patience: int = 15,
        random_state: int = 42,
        device: str = "auto",
    ):
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.lambda_sparse = lambda_sparse
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state

        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Will be set during fit
        self.model = None
        self.feature_names: Optional[List[str]] = None
        self.n_features: Optional[int] = None
        self.use_library = TABNET_AVAILABLE

        # For feature scaling (only needed for native impl)
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

    def _preprocess(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Preprocess features with standardization."""
        X = np.asarray(X, dtype=np.float32)

        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)

        if not self.use_library:
            # Native implementation needs standardization
            if fit:
                self.mean = np.mean(X, axis=0)
                self.std = np.std(X, axis=0)
                self.std[self.std == 0] = 1.0

            X = (X - self.mean) / self.std

        return X

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Train the TabNet model.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            X_val: Optional validation features for early stopping
            y_val: Optional validation labels
        """
        # Store feature info
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
            X = X.values
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        X = self._preprocess(np.asarray(X, dtype=np.float32), fit=True)
        y = np.asarray(y, dtype=np.int64)

        self.n_features = X.shape[1]

        # Prepare validation data
        has_val = X_val is not None and y_val is not None
        if has_val:
            if hasattr(X_val, 'values'):
                X_val = X_val.values
            X_val = self._preprocess(np.asarray(X_val, dtype=np.float32), fit=False)
            y_val = np.asarray(y_val, dtype=np.int64)

        logger.info(f"Training TabNet on {self.device}")
        logger.info(f"  Features: {self.n_features}, Samples: {len(X)}")
        logger.info(f"  Using library: {self.use_library}")

        if self.use_library:
            self._fit_library(X, y, X_val, y_val, has_val)
        else:
            self._fit_native(X, y, X_val, y_val, has_val)

        return self

    def _fit_library(self, X, y, X_val, y_val, has_val):
        """Fit using pytorch-tabnet library."""
        self.model = TabNetClassifier(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            lambda_sparse=self.lambda_sparse,
            momentum=self.momentum,
            seed=self.random_state,
            device_name=self.device,
            verbose=0,
        )

        eval_set = [(X_val, y_val)] if has_val else None
        eval_name = ['val'] if has_val else None

        self.model.fit(
            X, y,
            eval_set=eval_set,
            eval_name=eval_name,
            max_epochs=self.epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            drop_last=False,
        )

        logger.info(f"  Training complete. Best epoch: {self.model.best_epoch}")

    def _fit_native(self, X, y, X_val, y_val, has_val):
        """Fit using simplified native implementation."""
        # Build a simpler attention-based network as fallback
        self.model = _SimpleTabNet(
            n_features=self.n_features,
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
        )

        device = torch.device(self.device)
        self.model.to(device)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if has_val:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                logits, _ = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch_X)

            train_loss /= len(X)

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_logits, _ = self.model(X_val_tensor)
                    val_loss = criterion(val_logits, y_val_tensor).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(f"  Early stopping at epoch {epoch + 1}")
                        break

                if (epoch + 1) % 10 == 0:
                    logger.info(f"  Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"  Epoch {epoch + 1}: train_loss={train_loss:.4f}")

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.model.to(device)

        logger.info(f"  Training complete. Best val_loss: {best_val_loss:.4f}")

    def predict_proba(self, X) -> np.ndarray:
        """
        Predict home team win probabilities.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Array of shape (n_samples,) with home win probabilities
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if hasattr(X, 'values'):
            X = X.values

        X = self._preprocess(np.asarray(X, dtype=np.float32), fit=False)

        if self.use_library:
            probs = self.model.predict_proba(X)
            return probs[:, 1]
        else:
            device = torch.device(self.device)
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

            self.model.eval()
            with torch.no_grad():
                logits, _ = self.model(X_tensor)
                probs = F.softmax(logits, dim=1)

            return probs[:, 1].cpu().numpy()

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None or self.feature_names is None:
            return None

        if self.use_library:
            importance = self.model.feature_importances_
        else:
            # For native implementation, use attention weights
            # This is a simplified approximation
            importance = np.ones(self.n_features) / self.n_features

        # Normalize
        importance = importance / importance.sum()

        return dict(zip(self.feature_names, importance))

    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'config': {
                'n_d': self.n_d,
                'n_a': self.n_a,
                'n_steps': self.n_steps,
                'gamma': self.gamma,
                'n_independent': self.n_independent,
                'n_shared': self.n_shared,
                'lambda_sparse': self.lambda_sparse,
                'momentum': self.momentum,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'patience': self.patience,
                'random_state': self.random_state,
                'device': self.device,
            },
            'n_features': self.n_features,
            'feature_names': self.feature_names,
            'mean': self.mean,
            'std': self.std,
            'use_library': self.use_library,
        }

        if self.use_library and self.model is not None:
            # Save tabnet model separately
            model_path = path.with_suffix('.tabnet')
            self.model.save_model(str(model_path))
            save_dict['model_path'] = str(model_path)
        elif self.model is not None:
            save_dict['model_state'] = self.model.state_dict()

        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

    @classmethod
    def load(cls, path: Path) -> 'TabNetModel':
        """Load model from disk."""
        path = Path(path)

        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        instance = cls(**save_dict['config'])
        instance.n_features = save_dict['n_features']
        instance.feature_names = save_dict['feature_names']
        instance.mean = save_dict['mean']
        instance.std = save_dict['std']
        instance.use_library = save_dict['use_library']

        if instance.use_library and 'model_path' in save_dict:
            instance.model = TabNetClassifier()
            instance.model.load_model(save_dict['model_path'])
        elif 'model_state' in save_dict and instance.n_features:
            instance.model = _SimpleTabNet(
                n_features=instance.n_features,
                n_d=instance.n_d,
                n_a=instance.n_a,
                n_steps=instance.n_steps,
            )
            instance.model.load_state_dict(save_dict['model_state'])
            instance.model.to(torch.device(instance.device))

        return instance


class _SimpleTabNet(nn.Module):
    """
    Simplified TabNet-like architecture for fallback.

    This is a basic attention-based network that captures some
    TabNet concepts without full sequential attention.
    """

    def __init__(
        self,
        n_features: int,
        n_d: int = 8,
        n_a: int = 8,
        n_steps: int = 3,
        n_classes: int = 2,
    ):
        super().__init__()

        self.n_features = n_features
        self.n_steps = n_steps
        self.n_d = n_d

        # Initial batch normalization
        self.bn = nn.BatchNorm1d(n_features)

        # Shared feature transformer
        self.shared = nn.Sequential(
            nn.Linear(n_features, n_d + n_a),
            nn.BatchNorm1d(n_d + n_a),
            nn.ReLU(),
        )

        # Step-specific attention and decision layers
        self.attention_layers = nn.ModuleList()
        self.decision_layers = nn.ModuleList()

        for _ in range(n_steps):
            self.attention_layers.append(
                nn.Sequential(
                    nn.Linear(n_a, n_features),
                    nn.BatchNorm1d(n_features),
                )
            )
            self.decision_layers.append(
                nn.Sequential(
                    nn.Linear(n_d, n_d),
                    nn.BatchNorm1d(n_d),
                    nn.ReLU(),
                )
            )

        # Final classifier
        self.classifier = nn.Linear(n_d, n_classes)

    def forward(self, x):
        """Forward pass with attention aggregation."""
        batch_size = x.shape[0]

        # Normalize input
        x = self.bn(x)
        x_orig = x

        # Initialize
        prior_scales = torch.ones(batch_size, self.n_features, device=x.device)
        aggregated = torch.zeros(batch_size, self.n_d, device=x.device)
        total_entropy = 0.0

        for step in range(self.n_steps):
            # Masked features
            masked_x = x_orig * prior_scales

            # Shared transformation
            h = self.shared(masked_x)
            h_d, h_a = h[:, :self.n_d], h[:, self.n_d:]

            # Decision step output
            decision = self.decision_layers[step](h_d)
            aggregated = aggregated + decision

            # Attention for next step
            attention = self.attention_layers[step](h_a)
            attention = F.softmax(attention, dim=-1)

            # Update prior scales
            prior_scales = prior_scales * attention

            # Entropy for sparsity
            total_entropy += torch.mean(torch.sum(-attention * torch.log(attention + 1e-10), dim=-1))

        # Classification
        logits = self.classifier(aggregated)

        return logits, total_entropy
