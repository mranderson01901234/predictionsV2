"""
Probability Calibration

Provides calibration methods to improve the reliability of model probability outputs.
Supports:
- Platt Scaling (logistic regression on logits)
- Isotonic Regression (non-parametric monotonic calibration)

Usage:
    from models.calibration import CalibratedModel, PlattScaler, IsotonicCalibrator

    # Wrap any model with calibration
    calibrated = CalibratedModel(base_model, method="platt")
    calibrated.fit_calibration(X_val, y_val)
    probs = calibrated.predict_proba(X_test)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from typing import Optional, Literal, Union, Tuple
from pathlib import Path
import pickle
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.base import BaseModel

logger = logging.getLogger(__name__)


class PlattScaler:
    """
    Platt Scaling for probability calibration.

    Fits a logistic regression on the raw probability outputs
    to produce calibrated probabilities.
    """

    def __init__(self):
        self.calibrator = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
        self.fitted = False

    def fit(self, probs: np.ndarray, y: np.ndarray) -> 'PlattScaler':
        """
        Fit the Platt scaler.

        Args:
            probs: Raw probabilities from base model (n_samples,)
            y: True binary labels (n_samples,)
        """
        # Reshape for sklearn
        probs_reshaped = probs.reshape(-1, 1)

        # Fit logistic regression
        self.calibrator.fit(probs_reshaped, y)
        self.fitted = True

        logger.debug(f"Platt scaling: coef={self.calibrator.coef_[0][0]:.4f}, "
                     f"intercept={self.calibrator.intercept_[0]:.4f}")

        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling to probabilities.

        Args:
            probs: Raw probabilities (n_samples,)

        Returns:
            Calibrated probabilities (n_samples,)
        """
        if not self.fitted:
            raise ValueError("PlattScaler not fitted. Call fit() first.")

        probs_reshaped = probs.reshape(-1, 1)
        return self.calibrator.predict_proba(probs_reshaped)[:, 1]


class IsotonicCalibrator:
    """
    Isotonic Regression for probability calibration.

    Non-parametric method that ensures monotonicity.
    Generally more flexible than Platt scaling.
    """

    def __init__(self, out_of_bounds: str = 'clip'):
        self.calibrator = IsotonicRegression(out_of_bounds=out_of_bounds)
        self.fitted = False

    def fit(self, probs: np.ndarray, y: np.ndarray) -> 'IsotonicCalibrator':
        """
        Fit the isotonic calibrator.

        Args:
            probs: Raw probabilities from base model (n_samples,)
            y: True binary labels (n_samples,)
        """
        self.calibrator.fit(probs, y)
        self.fitted = True
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply isotonic calibration to probabilities.

        Args:
            probs: Raw probabilities (n_samples,)

        Returns:
            Calibrated probabilities (n_samples,)
        """
        if not self.fitted:
            raise ValueError("IsotonicCalibrator not fitted. Call fit() first.")

        return self.calibrator.predict(probs)


class CalibratedModel(BaseModel):
    """
    Wrapper that adds calibration to any base model.

    This preserves the original model while adding a calibration layer
    that can be trained on held-out validation data.

    Args:
        base_model: The underlying prediction model
        method: Calibration method ("platt" or "isotonic")
    """

    def __init__(
        self,
        base_model: Optional[BaseModel] = None,
        method: Literal["platt", "isotonic"] = "platt",
    ):
        self.base_model = base_model
        self.method = method

        if method == "platt":
            self.calibrator = PlattScaler()
        elif method == "isotonic":
            self.calibrator = IsotonicCalibrator()
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        self.calibration_fitted = False

    def fit(self, X, y, X_cal=None, y_cal=None):
        """
        Train base model and optionally fit calibration.

        Args:
            X: Training features
            y: Training labels
            X_cal: Optional calibration features (if None, uses X)
            y_cal: Optional calibration labels (if None, uses y)
        """
        # Train base model
        self.base_model.fit(X, y)

        # Fit calibration if calibration data provided
        if X_cal is not None and y_cal is not None:
            self.fit_calibration(X_cal, y_cal)
        else:
            # Use training data for calibration (not ideal but works)
            self.fit_calibration(X, y)

        return self

    def fit_calibration(self, X, y) -> 'CalibratedModel':
        """
        Fit the calibration layer on held-out data.

        Should be called after base model is trained, using
        validation data that wasn't used for training.

        Args:
            X: Calibration features
            y: Calibration labels
        """
        if self.base_model is None:
            raise ValueError("No base model set")

        # Get base model predictions
        raw_probs = self.base_model.predict_proba(X)
        y_arr = np.asarray(y)

        # Fit calibrator
        self.calibrator.fit(raw_probs, y_arr)
        self.calibration_fitted = True

        logger.info(f"Calibration fitted using {self.method} method on {len(y)} samples")

        return self

    def predict_proba(self, X, calibrated: bool = True) -> np.ndarray:
        """
        Predict probabilities.

        Args:
            X: Feature matrix
            calibrated: If True, return calibrated probabilities.
                        If False, return raw base model probabilities.

        Returns:
            Probability array
        """
        if self.base_model is None:
            raise ValueError("No base model set")

        raw_probs = self.base_model.predict_proba(X)

        if calibrated and self.calibration_fitted:
            return self.calibrator.transform(raw_probs)
        else:
            return raw_probs

    def predict_proba_raw(self, X) -> np.ndarray:
        """Get raw (uncalibrated) probabilities."""
        return self.predict_proba(X, calibrated=False)

    def predict_proba_both(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get both raw and calibrated probabilities.

        Returns:
            Tuple of (raw_probs, calibrated_probs)
        """
        raw = self.predict_proba(X, calibrated=False)
        calibrated = self.predict_proba(X, calibrated=True)
        return raw, calibrated

    def save(self, path: Path):
        """Save calibrated model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save base model separately
        base_model_path = path.parent / f"{path.stem}_base.pkl"
        if self.base_model is not None:
            self.base_model.save(base_model_path)

        save_dict = {
            'method': self.method,
            'calibrator': self.calibrator,
            'calibration_fitted': self.calibration_fitted,
            'base_model_path': str(base_model_path),
        }

        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

    @classmethod
    def load(cls, path: Path, base_model_loader=None) -> 'CalibratedModel':
        """Load calibrated model from disk."""
        path = Path(path)

        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        instance = cls(method=save_dict['method'])
        instance.calibrator = save_dict['calibrator']
        instance.calibration_fitted = save_dict['calibration_fitted']

        # Load base model
        loader = base_model_loader or BaseModel.load
        instance.base_model = loader(save_dict['base_model_path'])

        return instance


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute calibration metrics for a set of predictions.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration curve

    Returns:
        Dictionary with calibration metrics
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    # Expected Calibration Error (ECE)
    bin_sizes = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
    bin_sizes = bin_sizes / len(y_prob)
    ece = np.sum(bin_sizes[:len(prob_true)] * np.abs(prob_true - prob_pred))

    # Maximum Calibration Error (MCE)
    mce = np.max(np.abs(prob_true - prob_pred)) if len(prob_true) > 0 else 0.0

    # Brier score
    brier = np.mean((y_prob - y_true) ** 2)

    return {
        'ece': ece,
        'mce': mce,
        'brier': brier,
        'calibration_curve': (prob_true, prob_pred),
        'n_bins': n_bins,
    }


def calibrate_probabilities(
    probs: np.ndarray,
    y_cal: np.ndarray,
    method: Literal["platt", "isotonic"] = "platt",
) -> Tuple[np.ndarray, Union[PlattScaler, IsotonicCalibrator]]:
    """
    Convenience function to calibrate probabilities.

    Args:
        probs: Raw probabilities to calibrate
        y_cal: True labels for calibration
        method: Calibration method

    Returns:
        Tuple of (calibrated_probs, fitted_calibrator)
    """
    if method == "platt":
        calibrator = PlattScaler()
    else:
        calibrator = IsotonicCalibrator()

    calibrator.fit(probs, y_cal)
    calibrated = calibrator.transform(probs)

    return calibrated, calibrator
