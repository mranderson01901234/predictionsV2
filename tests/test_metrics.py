"""
Tests for Evaluation Metrics
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.metrics import accuracy, brier_score, log_loss, calibration_buckets


class TestAccuracy:
    """Test accuracy calculation."""
    
    def test_perfect_accuracy(self):
        """Test accuracy with perfect predictions."""
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 1])
        assert accuracy(y_true, y_pred) == 1.0
    
    def test_zero_accuracy(self):
        """Test accuracy with all wrong predictions."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1])
        assert accuracy(y_true, y_pred) == 0.0
    
    def test_partial_accuracy(self):
        """Test accuracy with partial correctness."""
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 0, 0, 1])
        assert accuracy(y_true, y_pred) == 0.8


class TestBrierScore:
    """Test Brier score calculation."""
    
    def test_perfect_predictions(self):
        """Test Brier score with perfect probabilities."""
        y_true = np.array([1, 0, 1, 0])
        p_pred = np.array([1.0, 0.0, 1.0, 0.0])
        assert brier_score(y_true, p_pred) == 0.0
    
    def test_worst_predictions(self):
        """Test Brier score with worst probabilities."""
        y_true = np.array([1, 0])
        p_pred = np.array([0.0, 1.0])
        assert brier_score(y_true, p_pred) == 1.0
    
    def test_intermediate_predictions(self):
        """Test Brier score with intermediate probabilities."""
        y_true = np.array([1, 0])
        p_pred = np.array([0.5, 0.5])
        # Brier = mean((0.5-1)^2 + (0.5-0)^2) = mean(0.25 + 0.25) = 0.25
        assert abs(brier_score(y_true, p_pred) - 0.25) < 1e-10


class TestLogLoss:
    """Test log loss calculation."""
    
    def test_perfect_predictions(self):
        """Test log loss with perfect probabilities."""
        y_true = np.array([1, 0])
        p_pred = np.array([1.0, 0.0])
        # Should be very small (close to 0)
        loss = log_loss(y_true, p_pred)
        assert loss < 1e-10
    
    def test_worst_predictions(self):
        """Test log loss with worst probabilities."""
        y_true = np.array([1, 0])
        p_pred = np.array([0.0, 1.0])
        # Should be very large
        loss = log_loss(y_true, p_pred)
        assert loss > 10  # Should be large
    
    def test_intermediate_predictions(self):
        """Test log loss with 0.5 probabilities."""
        y_true = np.array([1, 0])
        p_pred = np.array([0.5, 0.5])
        # log_loss = -mean(y*log(p) + (1-y)*log(1-p))
        # = -mean(1*log(0.5) + 0*log(0.5) + 0*log(0.5) + 1*log(0.5))
        # = -mean(log(0.5) + log(0.5)) = -log(0.5) = log(2) ≈ 0.693
        loss = log_loss(y_true, p_pred)
        expected = np.log(2)
        assert abs(loss - expected) < 1e-10


class TestCalibrationBuckets:
    """Test calibration bucket calculation."""
    
    def test_calibration_buckets(self):
        """Test that calibration buckets are created correctly."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        p_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        calib_df = calibration_buckets(y_true, p_pred, n_bins=5)
        
        assert len(calib_df) > 0
        assert "predicted_freq" in calib_df.columns
        assert "actual_freq" in calib_df.columns
        assert "count" in calib_df.columns
    
    def test_perfect_calibration(self):
        """Test with perfectly calibrated predictions."""
        # If predictions are perfectly calibrated, predicted_freq ≈ actual_freq
        y_true = np.array([1, 0, 1, 0, 1, 0])
        p_pred = np.array([0.8, 0.2, 0.8, 0.2, 0.8, 0.2])
        
        calib_df = calibration_buckets(y_true, p_pred, n_bins=3)
        
        # Check that calibration errors are small
        max_error = calib_df["calibration_error"].max()
        assert max_error < 0.5  # Allow some tolerance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

