"""
Report Generation

Generates markdown reports summarizing model evaluation results.
"""

from pathlib import Path
from typing import Dict
import pandas as pd
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_version import get_version_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_report(
    results: Dict,
    output_path: Path,
    train_seasons: list,
    validation_season: int,
    test_season: int,
) -> None:
    """
    Generate markdown evaluation report.
    
    Args:
        results: Dictionary with evaluation results (from run_backtest)
        output_path: Path to save report
        train_seasons: List of training seasons
        validation_season: Validation season
        test_season: Test season
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating report: {output_path}")
    
    # Get version info
    version_info = get_version_info(Path(__file__).parent.parent.parent)
    
    lines = []
    lines.append("# NFL Baseline Model Evaluation Report (Phase 1C)")
    lines.append("")
    lines.append("## Version Information")
    lines.append("")
    lines.append(f"- **Model Version**: {version_info['model_version']}")
    lines.append(f"- **Git SHA**: {version_info['git_sha']}")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- **Training Seasons**: {train_seasons}")
    lines.append(f"- **Validation Season**: {validation_season}")
    lines.append(f"- **Test Season**: {test_season}")
    lines.append("")
    lines.append("## Models Evaluated")
    lines.append("")
    lines.append("1. Logistic Regression")
    lines.append("2. Gradient Boosting")
    lines.append("3. Ensemble (blended)")
    lines.append("")
    
    # Model results
    for model_name, model_results in results.items():
        model_display_name = model_name.replace("_", " ").title()
        lines.append(f"## {model_display_name}")
        lines.append("")
        
        # Validation results
        val = model_results["validation"]
        lines.append("### Validation Set")
        lines.append("")
        lines.append(f"- **Games**: {val['n_games']}")
        lines.append(f"- **Accuracy**: {val['accuracy']:.4f}")
        lines.append(f"- **Brier Score**: {val['brier_score']:.4f}")
        lines.append(f"- **Log Loss**: {val['log_loss']:.4f}")
        lines.append(f"- **Mean Calibration Error**: {val['mean_calibration_error']:.4f}")
        lines.append("")
        
        # Calibration table
        lines.append("#### Calibration")
        lines.append("")
        calib_df = val["calibration_buckets"]
        lines.append("| Bin | Predicted | Actual | Count | Error |")
        lines.append("|-----|-----------|--------|-------|-------|")
        for _, row in calib_df.iterrows():
            lines.append(
                f"| {row['bin']} | {row['predicted_freq']:.3f} | "
                f"{row['actual_freq']:.3f} | {row['count']} | "
                f"{row['calibration_error']:.3f} |"
            )
        lines.append("")
        
        # ROI results
        lines.append("#### ROI vs Closing Line")
        lines.append("")
        for roi_key, roi_data in val["roi_results"].items():
            threshold = roi_data["edge_threshold"]
            lines.append(f"**Edge Threshold: {threshold:.0%}**")
            lines.append(f"- Bets: {roi_data['n_bets']}")
            lines.append(f"- Win Rate: {roi_data['win_rate']:.2%}")
            lines.append(f"- ROI: {roi_data['roi']:.2%}")
            lines.append("")
        
        # Test results
        test = model_results["test"]
        lines.append("### Test Set")
        lines.append("")
        lines.append(f"- **Games**: {test['n_games']}")
        lines.append(f"- **Accuracy**: {test['accuracy']:.4f}")
        lines.append(f"- **Brier Score**: {test['brier_score']:.4f}")
        lines.append(f"- **Log Loss**: {test['log_loss']:.4f}")
        lines.append(f"- **Mean Calibration Error**: {test['mean_calibration_error']:.4f}")
        lines.append("")
        
        # ROI results
        lines.append("#### ROI vs Closing Line")
        lines.append("")
        for roi_key, roi_data in test["roi_results"].items():
            threshold = roi_data["edge_threshold"]
            lines.append(f"**Edge Threshold: {threshold:.0%}**")
            lines.append(f"- Bets: {roi_data['n_bets']}")
            lines.append(f"- Win Rate: {roi_data['win_rate']:.2%}")
            lines.append(f"- ROI: {roi_data['roi']:.2%}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    # Summary comparison
    lines.append("## Model Comparison")
    lines.append("")
    lines.append("### Test Set Performance")
    lines.append("")
    lines.append("| Model | Accuracy | Brier | Log Loss | Calibration Error |")
    lines.append("|-------|----------|-------|----------|-------------------|")
    
    for model_name, model_results in results.items():
        test = model_results["test"]
        model_display = model_name.replace("_", " ").title()
        lines.append(
            f"| {model_display} | {test['accuracy']:.4f} | "
            f"{test['brier_score']:.4f} | {test['log_loss']:.4f} | "
            f"{test['mean_calibration_error']:.4f} |"
        )
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by Phase 1C evaluation pipeline*")
    
    # Write report
    content = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(content)
    
    logger.info(f"Report saved to {output_path}")


def generate_phase1d_report(
    results: Dict,
    season_analysis: pd.DataFrame,
    output_path: Path,
    train_seasons: list,
    validation_season: int,
    test_season: int,
) -> None:
    """
    Generate Phase 1D report with market baseline comparison and season-by-season analysis.
    
    Args:
        results: Dictionary with evaluation results (from run_backtest)
        season_analysis: DataFrame with season-by-season metrics
        output_path: Path to save report
        train_seasons: List of training seasons
        validation_season: Validation season
        test_season: Test season
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating Phase 1D report: {output_path}")
    
    # Get version info
    version_info = get_version_info(Path(__file__).parent.parent.parent)
    
    lines = []
    lines.append("# NFL Baseline Model Evaluation Report (Phase 1D)")
    lines.append("")
    lines.append("## Version Information")
    lines.append("")
    lines.append(f"- **Model Version**: {version_info['model_version']}")
    lines.append(f"- **Git SHA**: {version_info['git_sha']}")
    lines.append("")
    lines.append("## Sanity Check & Market Baseline Comparison")
    lines.append("")
    lines.append("This report validates the Phase 1C baseline model against a market-only benchmark")
    lines.append("and checks stability across seasons.")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- **Training Seasons**: {train_seasons}")
    lines.append(f"- **Validation Season**: {validation_season}")
    lines.append(f"- **Test Season**: {test_season}")
    lines.append("")
    
    # Model vs Market Comparison Table
    lines.append("## Model vs Market Baseline Comparison")
    lines.append("")
    lines.append("### Test Set Performance")
    lines.append("")
    lines.append("| Model | Accuracy | Brier | Log Loss | ROI (3%) | ROI (5%) |")
    lines.append("|-------|----------|-------|----------|---------|---------|")
    
    for model_name, model_results in results.items():
        test = model_results["test"]
        model_display = model_name.replace("_", " ").title()
        
        # Get ROI for 3% and 5% thresholds
        roi_3 = test["roi_results"].get("roi_threshold_0.03", {}).get("roi", 0.0)
        roi_5 = test["roi_results"].get("roi_threshold_0.05", {}).get("roi", 0.0)
        
        lines.append(
            f"| {model_display} | {test['accuracy']:.4f} | "
            f"{test['brier_score']:.4f} | {test['log_loss']:.4f} | "
            f"{roi_3:.2%} | {roi_5:.2%} |"
        )
    
    lines.append("")
    
    # Validation Set Comparison
    lines.append("### Validation Set Performance")
    lines.append("")
    lines.append("| Model | Accuracy | Brier | Log Loss | ROI (3%) | ROI (5%) |")
    lines.append("|-------|----------|-------|----------|---------|---------|")
    
    for model_name, model_results in results.items():
        val = model_results["validation"]
        model_display = model_name.replace("_", " ").title()
        
        roi_3 = val["roi_results"].get("roi_threshold_0.03", {}).get("roi", 0.0)
        roi_5 = val["roi_results"].get("roi_threshold_0.05", {}).get("roi", 0.0)
        
        lines.append(
            f"| {model_display} | {val['accuracy']:.4f} | "
            f"{val['brier_score']:.4f} | {val['log_loss']:.4f} | "
            f"{roi_3:.2%} | {roi_5:.2%} |"
        )
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Season-by-Season Analysis
    if len(season_analysis) > 0:
        lines.append("## Season-by-Season Stability Analysis")
        lines.append("")
        lines.append("Metrics for each season (model trained on previous seasons):")
        lines.append("")
        lines.append("| Season | Games | Model Acc | Model Brier | Model ROI | Market Acc | Market Brier | Market ROI |")
        lines.append("|--------|-------|-----------|-------------|-----------|------------|--------------|------------|")
        
        for _, row in season_analysis.iterrows():
            lines.append(
                f"| {row['season']} | {row['n_games']} | "
                f"{row['model_accuracy']:.3f} | {row['model_brier']:.3f} | "
                f"{row['model_roi']:.2%} | {row['market_accuracy']:.3f} | "
                f"{row['market_brier']:.3f} | {row['market_roi']:.2%} |"
            )
        
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # Detailed Model Results
    lines.append("## Detailed Model Results")
    lines.append("")
    
    for model_name, model_results in results.items():
        model_display_name = model_name.replace("_", " ").title()
        lines.append(f"### {model_display_name}")
        lines.append("")
        
        # Test results summary
        test = model_results["test"]
        lines.append("#### Test Set")
        lines.append("")
        lines.append(f"- **Games**: {test['n_games']}")
        lines.append(f"- **Accuracy**: {test['accuracy']:.4f}")
        lines.append(f"- **Brier Score**: {test['brier_score']:.4f}")
        lines.append(f"- **Log Loss**: {test['log_loss']:.4f}")
        lines.append("")
        
        # ROI details
        lines.append("##### ROI vs Closing Line")
        lines.append("")
        for roi_key, roi_data in test["roi_results"].items():
            threshold = roi_data["edge_threshold"]
            lines.append(f"**Edge Threshold: {threshold:.0%}**")
            lines.append(f"- Bets: {roi_data['n_bets']}")
            lines.append(f"- Win Rate: {roi_data['win_rate']:.2%}")
            lines.append(f"- ROI: {roi_data['roi']:.2%}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    # Key Findings
    lines.append("## Key Findings")
    lines.append("")
    
    # Compare model vs market on test set
    if "market_baseline" in results:
        market_test = results["market_baseline"]["test"]
        market_roi_3 = market_test["roi_results"].get("roi_threshold_0.03", {}).get("roi", 0.0)
        
        # Find best model ROI
        best_model_roi = 0.0
        best_model_name = ""
        for model_name, model_results in results.items():
            if model_name == "market_baseline":
                continue
            test_roi = model_results["test"]["roi_results"].get("roi_threshold_0.03", {}).get("roi", 0.0)
            if test_roi > best_model_roi:
                best_model_roi = test_roi
                best_model_name = model_name.replace("_", " ").title()
        
        lines.append(f"- **Market Baseline ROI (3% edge)**: {market_roi_3:.2%}")
        lines.append(f"- **Best Model ROI (3% edge)**: {best_model_roi:.2%} ({best_model_name})")
        
        if best_model_roi > market_roi_3:
            lines.append(f"- **Model Advantage**: {best_model_roi - market_roi_3:.2%} over market baseline")
        else:
            lines.append("- **Warning**: Model ROI is not better than market baseline")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by Phase 1D evaluation pipeline*")
    
    # Write report
    content = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(content)
    
    logger.info(f"Phase 1D report saved to {output_path}")


