"""
Calibration Validation Module

Provides tools for validating and visualizing calibration quality:
- Reliability diagrams (calibration curves)
- Accuracy by confidence tier
- Monotonicity checks
- Before/after calibration comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from models.calibration import compute_calibration_metrics, calibrate_probabilities

logger = logging.getLogger(__name__)


def calculate_accuracy_by_confidence(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    confidence_bins: List[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """
    Calculate accuracy for different confidence tiers.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        confidence_bins: List of (min, max) tuples for confidence bins.
                        If None, uses default bins: [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]
    
    Returns:
        DataFrame with columns: bin_min, bin_max, bin_midpoint, n_games, accuracy
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    # Calculate confidence (distance from 0.5)
    confidence = np.maximum(y_prob, 1 - y_prob)
    
    # Default bins if not provided
    if confidence_bins is None:
        confidence_bins = [
            (0.5, 0.6),
            (0.6, 0.7),
            (0.7, 0.8),
            (0.8, 1.0),
        ]
    
    results = []
    
    for bin_min, bin_max in confidence_bins:
        # Find games in this confidence bin
        mask = (confidence >= bin_min) & (confidence < bin_max)
        if bin_max >= 1.0:
            mask = (confidence >= bin_min) & (confidence <= bin_max)
        
        n_games = mask.sum()
        
        if n_games == 0:
            continue
        
        # Get predictions and actuals for this bin
        bin_y_true = y_true[mask]
        bin_y_prob = y_prob[mask]
        
        # Calculate accuracy (using 0.5 threshold)
        bin_y_pred = (bin_y_prob >= 0.5).astype(int)
        accuracy = (bin_y_pred == bin_y_true).mean()
        
        results.append({
            'bin_min': bin_min,
            'bin_max': bin_max,
            'bin_midpoint': (bin_min + bin_max) / 2,
            'n_games': n_games,
            'accuracy': accuracy,
        })
    
    return pd.DataFrame(results)


def check_monotonic_accuracy(
    accuracy_df: pd.DataFrame,
    tolerance: float = 0.03,
) -> Tuple[bool, List[str]]:
    """
    Check if accuracy is monotonic across confidence bins.
    
    Monotonic means: higher confidence → higher accuracy (within tolerance).
    
    Args:
        accuracy_df: DataFrame from calculate_accuracy_by_confidence
        tolerance: Tolerance for monotonicity (default 3%)
    
    Returns:
        Tuple of (is_monotonic, list of warnings)
    """
    warnings = []
    
    if len(accuracy_df) < 2:
        return True, []
    
    # Sort by bin midpoint
    accuracy_df = accuracy_df.sort_values('bin_midpoint')
    
    accuracies = accuracy_df['accuracy'].values
    bin_midpoints = accuracy_df['bin_midpoint'].values
    
    is_monotonic = True
    
    for i in range(1, len(accuracies)):
        prev_acc = accuracies[i-1]
        curr_acc = accuracies[i]
        prev_mid = bin_midpoints[i-1]
        curr_mid = bin_midpoints[i]
        
        # Expected: current accuracy should be >= previous (within tolerance)
        expected_min = prev_acc - tolerance
        
        if curr_acc < expected_min:
            is_monotonic = False
            warning = (
                f"Non-monotonic: {prev_mid:.1%} confidence → {prev_acc:.1%} accuracy, "
                f"{curr_mid:.1%} confidence → {curr_acc:.1%} accuracy "
                f"(expected ≥ {expected_min:.1%})"
            )
            warnings.append(warning)
    
    return is_monotonic, warnings


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
    output_path: Optional[Path] = None,
    show_plot: bool = False,
) -> plt.Figure:
    """
    Plot reliability diagram (calibration curve).
    
    Shows predicted probability vs actual frequency for each bin.
    Perfect calibration = points on diagonal line.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        title: Plot title
        output_path: Path to save figure (optional)
        show_plot: Whether to display plot
    
    Returns:
        Matplotlib figure
    """
    from sklearn.calibration import calibration_curve
    
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot calibration curve
    ax.plot(prob_pred, prob_true, 's-', label='Model', linewidth=2, markersize=8)
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=1.5)
    
    # Formatting
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved reliability diagram to {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def compare_calibration_methods(
    y_true: np.ndarray,
    y_prob_raw: np.ndarray,
    methods: List[str] = ['platt', 'isotonic', 'temperature'],
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Compare different calibration methods.
    
    Args:
        y_true: True binary labels
        y_prob_raw: Raw (uncalibrated) probabilities
        methods: List of calibration methods to compare
        output_dir: Directory to save plots and results
    
    Returns:
        Dictionary with comparison results
    """
    y_true = np.asarray(y_true)
    y_prob_raw = np.asarray(y_prob_raw)
    
    results = {}
    
    # Evaluate raw probabilities
    raw_metrics = compute_calibration_metrics(y_true, y_prob_raw)
    results['raw'] = {
        'ece': raw_metrics['ece'],
        'mce': raw_metrics['mce'],
        'brier': raw_metrics['brier'],
    }
    
    # Evaluate each calibration method
    for method in methods:
        try:
            y_prob_cal, calibrator = calibrate_probabilities(y_prob_raw, y_true, method=method)
            cal_metrics = compute_calibration_metrics(y_true, y_prob_cal)
            
            results[method] = {
                'ece': cal_metrics['ece'],
                'mce': cal_metrics['mce'],
                'brier': cal_metrics['brier'],
                'calibrator': calibrator,
            }
            
            logger.info(f"{method.capitalize()} calibration:")
            logger.info(f"  ECE: {cal_metrics['ece']:.4f}")
            logger.info(f"  MCE: {cal_metrics['mce']:.4f}")
            logger.info(f"  Brier: {cal_metrics['brier']:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to calibrate with {method}: {e}")
            results[method] = {'error': str(e)}
    
    # Create comparison plot
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot all methods together
        fig, ax = plt.subplots(figsize=(10, 8))
        
        from sklearn.calibration import calibration_curve
        
        # Plot raw
        prob_true, prob_pred = calibration_curve(y_true, y_prob_raw, n_bins=10)
        ax.plot(prob_pred, prob_true, 'o-', label='Raw', linewidth=2, markersize=6)
        
        # Plot each calibrated method
        for method in methods:
            if method in results and 'error' not in results[method]:
                y_prob_cal = results[method]['calibrator'].transform(y_prob_raw)
                prob_true, prob_pred = calibration_curve(y_true, y_prob_cal, n_bins=10)
                ax.plot(prob_pred, prob_true, 's-', label=method.capitalize(), linewidth=2, markersize=6)
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect', linewidth=1.5)
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title('Calibration Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        comparison_path = output_dir / "calibration_comparison.png"
        fig.savefig(comparison_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved calibration comparison to {comparison_path}")
        plt.close()
    
    return results


def validate_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    confidence_bins: List[Tuple[float, float]] = None,
    tolerance: float = 0.03,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Comprehensive calibration validation.
    
    Checks:
    1. Calibration metrics (ECE, MCE, Brier)
    2. Accuracy by confidence tier
    3. Monotonicity of accuracy
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        confidence_bins: Confidence bin definitions
        tolerance: Tolerance for monotonicity check
        output_dir: Directory to save plots and reports
    
    Returns:
        Dictionary with validation results
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    logger.info("Validating calibration...")
    
    # 1. Calibration metrics
    calib_metrics = compute_calibration_metrics(y_true, y_prob)
    
    # 2. Accuracy by confidence tier
    accuracy_by_conf = calculate_accuracy_by_confidence(y_true, y_prob, confidence_bins)
    
    # 3. Monotonicity check
    is_monotonic, warnings = check_monotonic_accuracy(accuracy_by_conf, tolerance)
    
    # Compile results
    results = {
        'calibration_metrics': {
            'ece': calib_metrics['ece'],
            'mce': calib_metrics['mce'],
            'brier': calib_metrics['brier'],
        },
        'accuracy_by_confidence': accuracy_by_conf.to_dict('records'),
        'monotonic': is_monotonic,
        'monotonicity_warnings': warnings,
    }
    
    # Log results
    logger.info("\nCalibration Metrics:")
    logger.info(f"  ECE: {calib_metrics['ece']:.4f}")
    logger.info(f"  MCE: {calib_metrics['mce']:.4f}")
    logger.info(f"  Brier: {calib_metrics['brier']:.4f}")
    
    logger.info("\nAccuracy by Confidence Tier:")
    for _, row in accuracy_by_conf.iterrows():
        logger.info(
            f"  {row['bin_min']:.0%}-{row['bin_max']:.0%}: "
            f"{row['accuracy']:.1%} ({row['n_games']} games)"
        )
    
    logger.info(f"\nMonotonic: {is_monotonic}")
    if warnings:
        logger.warning("Monotonicity warnings:")
        for warning in warnings:
            logger.warning(f"  - {warning}")
    
    # Generate plots
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Reliability diagram
        plot_reliability_diagram(
            y_true, y_prob,
            title="Calibration Reliability Diagram",
            output_path=output_dir / "reliability_diagram.png",
        )
        
        # Accuracy by confidence bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(
            range(len(accuracy_by_conf)),
            accuracy_by_conf['accuracy'],
            tick_label=[f"{row['bin_min']:.0%}-{row['bin_max']:.0%}" 
                       for _, row in accuracy_by_conf.iterrows()],
        )
        ax.axhline(y=0.5, color='r', linestyle='--', label='Coin Flip')
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_xlabel('Confidence Tier', fontsize=12)
        ax.set_title('Accuracy by Confidence Tier', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "accuracy_by_confidence.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    return results

