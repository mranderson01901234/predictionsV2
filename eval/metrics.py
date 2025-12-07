"""
Evaluation Metrics

Implements accuracy, Brier score, log loss, and calibration metrics.
"""

import numpy as np
from typing import Tuple
import pandas as pd


def accuracy(y_true: np.ndarray, y_pred_cls: np.ndarray) -> float:
    """
    Calculate accuracy.
    
    Args:
        y_true: True binary labels
        y_pred_cls: Predicted binary labels
    
    Returns:
        Accuracy score
    """
    return np.mean(y_true == y_pred_cls)


def brier_score(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    """
    Calculate Brier score (mean squared error of probabilities).
    
    Args:
        y_true: True binary labels
        p_pred: Predicted probabilities
    
    Returns:
        Brier score (lower is better)
    """
    return np.mean((p_pred - y_true) ** 2)


def log_loss(y_true: np.ndarray, p_pred: np.ndarray, eps: float = 1e-15) -> float:
    """
    Calculate log loss (binary cross-entropy).
    
    Args:
        y_true: True binary labels
        p_pred: Predicted probabilities
        eps: Small value to avoid log(0)
    
    Returns:
        Log loss (lower is better)
    """
    # Clip probabilities to avoid log(0)
    p_pred = np.clip(p_pred, eps, 1 - eps)
    
    return -np.mean(y_true * np.log(p_pred) + (1 - y_true) * np.log(1 - p_pred))


def calibration_buckets(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Calculate calibration buckets.
    
    Groups predictions into bins and compares predicted vs actual frequencies.
    
    Args:
        y_true: True binary labels
        p_pred: Predicted probabilities
        n_bins: Number of bins
    
    Returns:
        DataFrame with columns: bin, predicted_freq, actual_freq, count
    """
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(p_pred, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Calculate statistics per bin
    results = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        
        bin_pred_freq = p_pred[mask].mean()
        bin_actual_freq = y_true[mask].mean()
        count = mask.sum()
        
        results.append({
            "bin": i + 1,
            "bin_min": bin_edges[i],
            "bin_max": bin_edges[i + 1],
            "predicted_freq": bin_pred_freq,
            "actual_freq": bin_actual_freq,
            "count": count,
            "calibration_error": abs(bin_pred_freq - bin_actual_freq),
        })
    
    return pd.DataFrame(results)

