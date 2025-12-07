"""
FT-Transformer (Feature Tokenizer Transformer) for Tabular Data

Implementation based on "Revisiting Deep Learning Models for Tabular Data"
(Gorishniy et al., 2021).

Key ideas:
- Each feature is tokenized (embedded) independently
- A [CLS] token is prepended for classification
- Standard transformer encoder processes the sequence
- The [CLS] token representation is used for prediction
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


class NumericalEmbedding(nn.Module):
    """
    Embedding layer for numerical features.

    Each numerical feature gets its own linear projection to the embedding dimension.
    This is the "Feature Tokenizer" part of FT-Transformer.
    """

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # Each feature has its own weight and bias
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.empty(n_features, d_model))

        # Initialize
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_features) tensor of numerical values

        Returns:
            (batch_size, n_features, d_model) tensor of embeddings
        """
        # x: (batch, n_features) -> (batch, n_features, 1)
        x = x.unsqueeze(-1)
        # Broadcast multiply: (batch, n_features, 1) * (n_features, d_model)
        # Then add bias
        return x * self.weight + self.bias


class TransformerBlock(nn.Module):
    """
    Standard transformer encoder block with pre-normalization.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)

        # Pre-norm feedforward
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = x + ff_out

        return x


class FTTransformerNetwork(nn.Module):
    """
    FT-Transformer network for tabular classification.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        n_classes: int = 2,
    ):
        super().__init__()

        self.n_features = n_features
        self.d_model = d_model

        # Feature embeddings
        self.feature_embeddings = NumericalEmbedding(n_features, d_model)

        # CLS token (learnable)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final normalization and classification head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_features) tensor of numerical features

        Returns:
            (batch_size, n_classes) logits
        """
        batch_size = x.shape[0]

        # Embed features: (batch, n_features, d_model)
        x = self.feature_embeddings(x)

        # Prepend CLS token: (batch, 1 + n_features, d_model)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Take CLS token output
        cls_output = x[:, 0]  # (batch, d_model)

        # Normalize and classify
        cls_output = self.norm(cls_output)
        logits = self.head(cls_output)

        return logits


class FTTransformerModel(BaseModel):
    """
    FT-Transformer model for NFL game prediction.

    Wraps the FTTransformerNetwork with sklearn-like interface.

    Args:
        d_model: Dimension of transformer embeddings (default: 64)
        n_heads: Number of attention heads (default: 4)
        n_layers: Number of transformer blocks (default: 3)
        d_ff: Feedforward layer dimension (default: 256)
        dropout: Dropout rate (default: 0.1)
        learning_rate: Learning rate for Adam optimizer (default: 1e-4)
        weight_decay: L2 regularization (default: 1e-5)
        batch_size: Training batch size (default: 64)
        epochs: Number of training epochs (default: 100)
        patience: Early stopping patience (default: 10)
        random_state: Random seed for reproducibility (default: 42)
        device: Device to train on ("cuda", "cpu", or "auto")
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        batch_size: int = 64,
        epochs: int = 100,
        patience: int = 10,
        random_state: int = 42,
        device: str = "auto",
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state

        # Set device
        if device == "auto":
            # Check if CUDA is available and actually usable (not just detected)
            # Some GPUs (e.g., GTX 1050 with CUDA 6.1) may be detected but incompatible
            use_cuda = False
            if torch.cuda.is_available():
                try:
                    # Try to create a small tensor on CUDA to verify compatibility
                    test_tensor = torch.zeros(1).cuda()
                    # Try a simple operation to ensure CUDA works
                    _ = test_tensor + 1
                    use_cuda = True
                    del test_tensor
                    torch.cuda.empty_cache()
                except Exception:
                    # CUDA not usable (incompatible GPU, driver issues, etc.), fall back to CPU
                    use_cuda = False
                    logger.warning("CUDA detected but not usable, falling back to CPU")
            
            self.device = torch.device("cuda" if use_cuda else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")

        # Will be set during fit
        self.model: Optional[FTTransformerNetwork] = None
        self.feature_names: Optional[List[str]] = None
        self.n_features: Optional[int] = None

        # For feature scaling
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

    def _preprocess(self, X: np.ndarray, fit: bool = False) -> torch.Tensor:
        """Preprocess features with standardization."""
        X = np.asarray(X, dtype=np.float32)

        if fit:
            self.mean = np.nanmean(X, axis=0)
            self.std = np.nanstd(X, axis=0)
            self.std[self.std == 0] = 1.0  # Avoid division by zero

        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)

        # Standardize
        X = (X - self.mean) / self.std

        return torch.tensor(X, dtype=torch.float32)

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Train the FT-Transformer model.

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

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        self.n_features = X.shape[1]

        # Preprocess
        X_tensor = self._preprocess(X, fit=True).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Validation data
        has_val = X_val is not None and y_val is not None
        if has_val:
            if hasattr(X_val, 'values'):
                X_val = X_val.values
            X_val_tensor = self._preprocess(X_val, fit=False).to(self.device)
            y_val_tensor = torch.tensor(np.asarray(y_val), dtype=torch.long).to(self.device)

        # Initialize model
        self.model = FTTransformerNetwork(
            n_features=self.n_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
        ).to(self.device)

        # Optimizer and loss
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # Training loop with early stopping
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        logger.info(f"Training FT-Transformer on {self.device}")
        logger.info(f"  Features: {self.n_features}, Samples: {len(X)}")
        logger.info(f"  Model: d={self.d_model}, heads={self.n_heads}, layers={self.n_layers}")

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch_X)

            train_loss /= len(X)

            # Validation
            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(X_val_tensor)
                    val_loss = criterion(val_logits, y_val_tensor).item()

                scheduler.step(val_loss)

                # Early stopping
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

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.model.to(self.device)

        logger.info(f"  Training complete. Best val_loss: {best_val_loss:.4f}")

        return self

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

        X_tensor = self._preprocess(X, fit=False).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = F.softmax(logits, dim=1)

        # Return probability of class 1 (home win)
        return probs[:, 1].cpu().numpy()

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance based on embedding weights.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None or self.feature_names is None:
            return None

        # Use L2 norm of embedding weights as importance proxy
        with torch.no_grad():
            weights = self.model.feature_embeddings.weight.cpu().numpy()
            importance = np.linalg.norm(weights, axis=1)

        # Normalize
        importance = importance / importance.sum()

        return dict(zip(self.feature_names, importance))

    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model state and config
        save_dict = {
            'model_state': self.model.state_dict() if self.model else None,
            'config': {
                'd_model': self.d_model,
                'n_heads': self.n_heads,
                'n_layers': self.n_layers,
                'd_ff': self.d_ff,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'patience': self.patience,
                'random_state': self.random_state,
            },
            'n_features': self.n_features,
            'feature_names': self.feature_names,
            'mean': self.mean,
            'std': self.std,
        }

        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

    @classmethod
    def load(cls, path: Path) -> 'FTTransformerModel':
        """Load model from disk."""
        path = Path(path)

        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        # Create instance with saved config
        instance = cls(**save_dict['config'])
        instance.n_features = save_dict['n_features']
        instance.feature_names = save_dict['feature_names']
        instance.mean = save_dict['mean']
        instance.std = save_dict['std']

        # Rebuild and load model
        if save_dict['model_state'] is not None and instance.n_features:
            instance.model = FTTransformerNetwork(
                n_features=instance.n_features,
                d_model=instance.d_model,
                n_heads=instance.n_heads,
                n_layers=instance.n_layers,
                d_ff=instance.d_ff,
                dropout=instance.dropout,
            )
            instance.model.load_state_dict(save_dict['model_state'])
            instance.model.to(instance.device)

        return instance
