"""
Calibration Diagnosis and Improvement

Provides tools to diagnose and fix calibration issues in NFL prediction models.

Features:
- Detailed bin-by-bin calibration analysis
- Comparison of different calibration methods
- Identification of problematic confidence tiers
- Visualization support (matplotlib optional)

Usage:
    from models.calibration_diagnosis import (
        diagnose_calibration,
        compare_calibration_methods,
        get_best_calibration
    )

    # Diagnose current calibration
    diagnosis = diagnose_calibration(y_true, y_pred_proba, model_name="My Model")

    # Compare and select best calibration method
    results, best_method = compare_calibration_methods(y_true, y_pred_proba)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)


@dataclass
class CalibrationBin:
    """Statistics for a single calibration bin."""
    low: float
    high: float
    count: int
    actual_rate: float
    predicted_rate: float
    error: float
    contribution_to_ece: float
    status: str  # "good", "warning", "bad"


@dataclass
class CalibrationDiagnosis:
    """Full calibration diagnosis result."""
    model_name: str
    n_samples: int
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier: float
    bins: List[CalibrationBin]
    problematic_bins: List[CalibrationBin]
    recommendations: List[str]


def diagnose_calibration(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str = "Model",
    n_bins: int = 10,
    print_report: bool = True,
) -> CalibrationDiagnosis:
    """
    Comprehensive calibration diagnosis.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        model_name: Name for reporting
        n_bins: Number of calibration bins
        print_report: Whether to print detailed report

    Returns:
        CalibrationDiagnosis with detailed results
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    # Define bins
    bin_edges = [0.50, 0.52, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.01]

    bins = []
    total_ece = 0.0
    max_error = 0.0

    for i in range(len(bin_edges) - 1):
        low = bin_edges[i]
        high = bin_edges[i + 1]

        # Handle both sides of 0.5
        mask_high = (y_pred_proba >= low) & (y_pred_proba < high)
        mask_low = (y_pred_proba > (1 - high)) & (y_pred_proba <= (1 - low))
        mask = mask_high  # Focus on >0.5 for clarity

        count = mask.sum()

        if count > 0:
            actual = y_true[mask].mean()
            predicted = y_pred_proba[mask].mean()
            error = actual - predicted
            abs_error = abs(error)
            ece_contrib = (count / len(y_true)) * abs_error
            total_ece += ece_contrib
            max_error = max(max_error, abs_error)

            # Determine status
            if abs_error < 0.03:
                status = "good"
            elif abs_error < 0.07:
                status = "warning"
            else:
                status = "bad"
        else:
            actual = 0.0
            predicted = (low + high) / 2
            error = 0.0
            ece_contrib = 0.0
            status = "empty"

        bins.append(CalibrationBin(
            low=low,
            high=high,
            count=count,
            actual_rate=actual,
            predicted_rate=predicted,
            error=error,
            contribution_to_ece=ece_contrib,
            status=status,
        ))

    # Identify problematic bins
    problematic_bins = [b for b in bins if b.status == "bad"]

    # Brier score
    brier = np.mean((y_pred_proba - y_true) ** 2)

    # Generate recommendations
    recommendations = []
    if total_ece > 0.05:
        recommendations.append("High overall calibration error - consider recalibration")

    # Check 50-60% tier specifically (common problem area)
    tier_50_60 = [b for b in bins if b.low == 0.50 or b.low == 0.52 or b.low == 0.55]
    tier_50_60_error = np.mean([abs(b.error) for b in tier_50_60 if b.count > 0])
    if tier_50_60_error > 0.05:
        recommendations.append("Poor calibration in 50-60% confidence tier - this is common and often caused by overconfidence")

    if len(problematic_bins) > 2:
        recommendations.append(f"{len(problematic_bins)} bins have >7% calibration error - systematic recalibration needed")

    if any(b.error > 0.05 for b in bins if b.count > 0):
        recommendations.append("Some tiers show underconfidence (actual > predicted) - model may be too conservative")

    if any(b.error < -0.05 for b in bins if b.count > 0):
        recommendations.append("Some tiers show overconfidence (actual < predicted) - consider temperature scaling")

    diagnosis = CalibrationDiagnosis(
        model_name=model_name,
        n_samples=len(y_true),
        ece=total_ece,
        mce=max_error,
        brier=brier,
        bins=bins,
        problematic_bins=problematic_bins,
        recommendations=recommendations,
    )

    if print_report:
        print_calibration_report(diagnosis)

    return diagnosis


def print_calibration_report(diagnosis: CalibrationDiagnosis):
    """Print a formatted calibration report."""
    print(f"\n{'='*60}")
    print(f"CALIBRATION DIAGNOSIS: {diagnosis.model_name}")
    print(f"{'='*60}")
    print(f"\nSamples: {diagnosis.n_samples}")
    print(f"Expected Calibration Error (ECE): {diagnosis.ece:.4f}")
    print(f"Maximum Calibration Error (MCE): {diagnosis.mce:.4f}")
    print(f"Brier Score: {diagnosis.brier:.4f}")

    print(f"\n{'Bin':<12} {'Count':<8} {'Actual':<10} {'Predicted':<10} {'Error':<10} {'Status'}")
    print("-" * 60)

    for b in diagnosis.bins:
        if b.count > 0:
            status_emoji = {"good": "[OK]", "warning": "[!]", "bad": "[X]", "empty": "[ ]"}.get(b.status, "")
            print(f"{b.low:.2f}-{b.high:.2f}  {b.count:<8} {b.actual_rate:<10.3f} {b.predicted_rate:<10.3f} {b.error:+.3f}     {status_emoji}")

    if diagnosis.recommendations:
        print(f"\nRecommendations:")
        for rec in diagnosis.recommendations:
            print(f"  - {rec}")


def apply_platt_scaling(probs: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Any]:
    """Apply Platt scaling and return calibrated probabilities and calibrator."""
    calibrator = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    calibrator.fit(probs.reshape(-1, 1), y)
    calibrated = calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
    return calibrated, calibrator


def apply_isotonic_calibration(probs: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Any]:
    """Apply isotonic regression and return calibrated probabilities and calibrator."""
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(probs, y)
    calibrated = calibrator.predict(probs)
    return calibrated, calibrator


def apply_temperature_scaling(probs: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """Apply temperature scaling and return calibrated probabilities and temperature."""
    # Clip probabilities to avoid log(0)
    probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
    logits = np.log(probs_clipped / (1 - probs_clipped))

    def nll(temperature):
        """Negative log-likelihood for given temperature."""
        scaled_logits = logits / temperature
        scaled_probs = 1 / (1 + np.exp(-scaled_logits))
        scaled_probs = np.clip(scaled_probs, 1e-7, 1 - 1e-7)
        return -np.sum(y * np.log(scaled_probs) + (1 - y) * np.log(1 - scaled_probs))

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
    T = result.x

    scaled_logits = logits / T
    calibrated = 1 / (1 + np.exp(-scaled_logits))

    logger.info(f"Optimal temperature: {T:.3f}")

    return calibrated, T


def apply_histogram_binning(
    probs: np.ndarray,
    y: np.ndarray,
    n_bins: int = 15
) -> Tuple[np.ndarray, Dict]:
    """Apply histogram binning calibration."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_calibration = {}
    calibrated = np.zeros_like(probs)

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i+1])
        if mask.sum() > 0:
            bin_calibration[(bin_edges[i], bin_edges[i+1])] = y[mask].mean()
            calibrated[mask] = y[mask].mean()
        else:
            calibrated[mask] = (bin_edges[i] + bin_edges[i+1]) / 2

    return calibrated, bin_calibration


def compare_calibration_methods(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    methods: Optional[List[str]] = None,
    print_results: bool = True,
) -> Tuple[Dict[str, Dict], str]:
    """
    Compare different calibration methods and identify the best one.

    Args:
        y_true: True binary labels
        y_pred_proba: Raw predicted probabilities
        methods: List of methods to compare (default: ['platt', 'isotonic', 'temperature'])
        print_results: Whether to print comparison

    Returns:
        Tuple of (results_dict, best_method_name)
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    if methods is None:
        methods = ['platt', 'isotonic', 'temperature']

    if print_results:
        print("\n" + "="*60)
        print("CALIBRATION METHOD COMPARISON")
        print("="*60)

    results = {}

    for method in methods:
        try:
            if method == 'platt':
                calibrated, calibrator = apply_platt_scaling(y_pred_proba, y_true)
            elif method == 'isotonic':
                calibrated, calibrator = apply_isotonic_calibration(y_pred_proba, y_true)
            elif method == 'temperature':
                calibrated, calibrator = apply_temperature_scaling(y_pred_proba, y_true)
            elif method == 'histogram':
                calibrated, calibrator = apply_histogram_binning(y_pred_proba, y_true)
            else:
                logger.warning(f"Unknown calibration method: {method}")
                continue

            # Compute metrics on calibrated probabilities
            diagnosis = diagnose_calibration(
                y_true, calibrated,
                model_name=f"Calibrated ({method})",
                print_report=False
            )

            results[method] = {
                'calibrated_proba': calibrated,
                'calibrator': calibrator,
                'ece': diagnosis.ece,
                'mce': diagnosis.mce,
                'brier': diagnosis.brier,
            }

            if print_results:
                print(f"\n{method.upper()}:")
                print(f"  ECE: {diagnosis.ece:.4f}")
                print(f"  MCE: {diagnosis.mce:.4f}")
                print(f"  Brier: {diagnosis.brier:.4f}")

        except Exception as e:
            logger.error(f"Error with {method} calibration: {e}")
            results[method] = {'error': str(e)}

    # Find best method by ECE
    valid_methods = {k: v for k, v in results.items() if 'ece' in v}
    if not valid_methods:
        raise ValueError("No calibration methods succeeded")

    best_method = min(valid_methods.keys(), key=lambda x: results[x]['ece'])

    if print_results:
        print(f"\n>>> Best calibration method: {best_method} (ECE: {results[best_method]['ece']:.4f})")

    return results, best_method


def get_best_calibration(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    methods: Optional[List[str]] = None,
) -> Tuple[np.ndarray, str, Any]:
    """
    Get the best-calibrated probabilities.

    Args:
        y_true: True binary labels
        y_pred_proba: Raw predicted probabilities
        methods: Methods to compare

    Returns:
        Tuple of (calibrated_probs, method_name, calibrator)
    """
    results, best_method = compare_calibration_methods(
        y_true, y_pred_proba, methods, print_results=False
    )

    return (
        results[best_method]['calibrated_proba'],
        best_method,
        results[best_method]['calibrator']
    )


def fix_confidence_tier(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    tier_low: float,
    tier_high: float,
) -> np.ndarray:
    """
    Fix calibration for a specific confidence tier.

    This uses local isotonic regression to fix calibration
    in a problematic tier while preserving other tiers.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        tier_low: Lower bound of tier
        tier_high: Upper bound of tier

    Returns:
        Adjusted probabilities
    """
    mask = (y_pred_proba >= tier_low) & (y_pred_proba < tier_high)

    if mask.sum() < 10:
        return y_pred_proba

    # Fit local calibration
    local_probs = y_pred_proba[mask]
    local_y = y_true[mask]

    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(local_probs, local_y)

    # Apply only to this tier
    result = y_pred_proba.copy()
    result[mask] = calibrator.predict(local_probs)

    return result


def compute_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration curve data.

    Returns:
        Tuple of (bin_centers, actual_rates, predicted_rates)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    actual_rates = []
    predicted_rates = []

    for i in range(n_bins):
        mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i+1])
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
            actual_rates.append(y_true[mask].mean())
            predicted_rates.append(y_pred_proba[mask].mean())

    return np.array(bin_centers), np.array(actual_rates), np.array(predicted_rates)


def analyze_50_60_tier(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> Dict[str, Any]:
    """
    Detailed analysis of the problematic 50-60% confidence tier.

    This tier is often the most problematic because:
    - Small edges mean noise dominates
    - Overconfidence is common here
    - This is where most bets would be placed

    Returns:
        Dictionary with detailed tier analysis
    """
    # Define sub-bins within 50-60%
    sub_bins = [(0.50, 0.52), (0.52, 0.54), (0.54, 0.56), (0.56, 0.58), (0.58, 0.60)]

    analysis = {
        'sub_bins': [],
        'overall': {},
        'recommendations': []
    }

    overall_mask = (y_pred_proba >= 0.50) & (y_pred_proba < 0.60)
    if overall_mask.sum() > 0:
        overall_actual = y_true[overall_mask].mean()
        overall_predicted = y_pred_proba[overall_mask].mean()
        analysis['overall'] = {
            'count': overall_mask.sum(),
            'actual': overall_actual,
            'predicted': overall_predicted,
            'error': overall_actual - overall_predicted,
        }

    for low, high in sub_bins:
        mask = (y_pred_proba >= low) & (y_pred_proba < high)
        if mask.sum() > 0:
            actual = y_true[mask].mean()
            predicted = y_pred_proba[mask].mean()
            error = actual - predicted

            analysis['sub_bins'].append({
                'range': f"{low:.2f}-{high:.2f}",
                'count': mask.sum(),
                'actual': actual,
                'predicted': predicted,
                'error': error,
            })

    # Generate specific recommendations
    if analysis['overall'].get('error', 0) < -0.03:
        analysis['recommendations'].append(
            "This tier is overconfident - predicted probabilities are too high. "
            "Consider temperature scaling (T > 1) to make predictions more conservative."
        )
    elif analysis['overall'].get('error', 0) > 0.03:
        analysis['recommendations'].append(
            "This tier is underconfident - actual win rates are higher than predicted. "
            "Consider temperature scaling (T < 1) or isotonic calibration."
        )

    return analysis


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Simulate some predictions
    n = 500
    y_true = np.random.binomial(1, 0.52, n)  # Slight home advantage
    y_pred_raw = 0.5 + 0.15 * (y_true - 0.5) + np.random.normal(0, 0.1, n)
    y_pred_raw = np.clip(y_pred_raw, 0.01, 0.99)

    # Add some miscalibration in 50-60% tier
    mask = (y_pred_raw >= 0.50) & (y_pred_raw < 0.60)
    y_pred_raw[mask] += 0.05  # Make overconfident

    # Diagnose
    diagnosis = diagnose_calibration(y_true, y_pred_raw, "Raw Model")

    # Compare methods
    results, best = compare_calibration_methods(y_true, y_pred_raw)

    # Analyze 50-60% tier
    print("\n" + "="*60)
    print("50-60% TIER ANALYSIS")
    print("="*60)
    tier_analysis = analyze_50_60_tier(y_true, y_pred_raw)
    for sub in tier_analysis['sub_bins']:
        print(f"  {sub['range']}: n={sub['count']}, actual={sub['actual']:.3f}, "
              f"predicted={sub['predicted']:.3f}, error={sub['error']:+.3f}")
    for rec in tier_analysis['recommendations']:
        print(f"\n  Recommendation: {rec}")
