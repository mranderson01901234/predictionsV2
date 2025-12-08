"""
Full Backtest Simulation: 2019-2022 Seasons

Simulates predictions for all games in 2019-2022 seasons and compares
against actual results. Generates comprehensive metrics and analysis.

Usage:
    python scripts/simulate_2019_2022_backtest.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import json
from typing import Dict, List, Tuple
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.training.trainer import load_features, load_backtest_config
from models.evaluation.evaluate import load_model, smart_load_base_model
from models.architectures.market_baseline import MarketBaselineModel
from eval.metrics import accuracy, brier_score, log_loss, calibration_buckets
from eval.backtest import (
    compute_market_implied_probabilities,
    calculate_roi,
    evaluate_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def simulate_seasons(
    model,
    seasons: List[int],
    feature_table: str = "baseline",
    model_path: Path = None,
) -> pd.DataFrame:
    """
    Simulate predictions for multiple seasons.
    
    Args:
        model: Trained model
        seasons: List of seasons to simulate
        feature_table: Feature table name
        model_path: Path to model (for logging)
    
    Returns:
        DataFrame with predictions and actual results
    """
    logger.info("=" * 80)
    logger.info(f"Full Backtest Simulation: Seasons {seasons}")
    logger.info("=" * 80)
    
    # Load features
    logger.info(f"\n[Step 1/5] Loading features from table: {feature_table}")
    X, y, feature_cols, df = load_features(feature_table=feature_table)
    logger.info(f"✓ Loaded {len(X)} total games")
    
    # Filter to requested seasons
    season_mask = df["season"].isin(seasons)
    X_seasons = X[season_mask].copy()
    y_seasons = y[season_mask].copy()
    df_seasons = df[season_mask].copy()
    
    logger.info(f"✓ Filtered to {len(X_seasons)} games in seasons {seasons}")
    
    # Check data availability
    available_seasons = sorted(df_seasons['season'].unique())
    logger.info(f"Available seasons in data: {available_seasons}")
    
    missing_seasons = [s for s in seasons if s not in available_seasons]
    if missing_seasons:
        logger.warning(f"Missing seasons: {missing_seasons}")
    
    # Make predictions
    logger.info(f"\n[Step 2/5] Making predictions...")
    y_pred_proba = model.predict_proba(X_seasons)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    logger.info(f"✓ Generated predictions for {len(X_seasons)} games")
    
    # Create results dataframe
    results_df = df_seasons.copy()
    results_df['predicted_prob'] = y_pred_proba
    results_df['predicted_winner'] = results_df.apply(
        lambda row: row['home_team'] if row['predicted_prob'] >= 0.5 else row['away_team'],
        axis=1
    )
    results_df['actual_winner'] = results_df.apply(
        lambda row: row['home_team'] if row['home_score'] > row['away_score'] else row['away_team'],
        axis=1
    )
    results_df['correct'] = (results_df['predicted_winner'] == results_df['actual_winner'])
    results_df['confidence'] = results_df['predicted_prob'].apply(
        lambda p: max(p, 1 - p)
    )
    
    # Calculate spread predictions (simple approximation)
    # Spread ≈ (p - 0.5) * 14 (roughly, a 50% prob = 0 spread, 100% = 14 point favorite)
    results_df['predicted_spread'] = (results_df['predicted_prob'] - 0.5) * 14
    results_df['actual_spread'] = results_df['home_score'] - results_df['away_score']
    results_df['spread_error'] = abs(results_df['predicted_spread'] - results_df['actual_spread'])
    
    logger.info(f"✓ Created results dataframe")
    
    return results_df


def analyze_results(
    results_df: pd.DataFrame,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
) -> Dict:
    """
    Analyze prediction results and calculate comprehensive metrics.
    
    Args:
        results_df: DataFrame with predictions and actuals
        model: Trained model
        X: Feature matrix
        y: Target vector
        df: Full dataframe
    
    Returns:
        Dictionary with analysis metrics
    """
    logger.info(f"\n[Step 3/5] Analyzing results...")
    
    # Overall metrics
    total_games = len(results_df)
    correct_predictions = results_df['correct'].sum()
    overall_accuracy = correct_predictions / total_games
    
    # Calculate Brier score and log loss
    y_true = results_df['home_win'].values
    y_proba = results_df['predicted_prob'].values
    brier = brier_score(y_true, y_proba)
    logloss = log_loss(y_true, y_proba)
    
    # Spread error metrics
    spread_errors = results_df['spread_error'].dropna()
    mean_spread_error = spread_errors.mean()
    median_spread_error = spread_errors.median()
    std_spread_error = spread_errors.std()
    rmse_spread = np.sqrt((spread_errors ** 2).mean())
    
    # Market comparison
    logger.info("Calculating market baseline comparison...")
    market_model = MarketBaselineModel()
    p_market = compute_market_implied_probabilities(results_df)
    
    # Market accuracy
    y_pred_market = (p_market >= 0.5).astype(int)
    market_accuracy = accuracy(y_true, y_pred_market)
    market_brier = brier_score(y_true, p_market.values)
    
    # ROI vs market
    edge_thresholds = [0.03, 0.05]
    roi_results = {}
    for edge_threshold in edge_thresholds:
        roi_dict = calculate_roi(
            y_true,
            y_proba,
            p_market.values,
            edge_threshold=edge_threshold,
        )
        roi_results[f"edge_{edge_threshold}"] = roi_dict
    
    # Season-by-season analysis
    logger.info("Calculating season-by-season metrics...")
    season_metrics = []
    for season in sorted(results_df['season'].unique()):
        season_df = results_df[results_df['season'] == season]
        season_y_true = season_df['home_win'].values
        season_y_proba = season_df['predicted_prob'].values
        season_y_pred = (season_y_proba >= 0.5).astype(int)
        
        season_acc = accuracy(season_y_true, season_y_pred)
        season_brier = brier_score(season_y_true, season_y_proba)
        season_correct = (season_df['predicted_winner'] == season_df['actual_winner']).sum()
        
        season_metrics.append({
            'season': season,
            'n_games': len(season_df),
            'accuracy': season_acc,
            'brier_score': season_brier,
            'correct_predictions': int(season_correct),
            'mean_spread_error': season_df['spread_error'].mean(),
        })
    
    # Week-by-week analysis
    week_metrics = []
    for week in sorted(results_df['week'].unique()):
        week_df = results_df[results_df['week'] == week]
        week_acc = week_df['correct'].mean()
        week_metrics.append({
            'week': week,
            'n_games': len(week_df),
            'accuracy': week_acc,
            'mean_spread_error': week_df['spread_error'].mean(),
        })
    
    # Confidence analysis
    results_df['confidence_bin'] = pd.cut(
        results_df['confidence'],
        bins=[0, 0.5, 0.6, 0.7, 0.8, 1.0],
        labels=['<50%', '50-60%', '60-70%', '70-80%', '80%+']
    )
    confidence_metrics = results_df.groupby('confidence_bin').agg({
        'correct': ['sum', 'count', 'mean'],
        'spread_error': 'mean'
    }).reset_index()
    confidence_metrics.columns = [
        'confidence_bin', 'correct', 'total', 'accuracy', 'mean_spread_error'
    ]
    
    # Calibration analysis
    cal_buckets_df = calibration_buckets(y_true, y_proba, n_bins=10)
    
    # Home vs away analysis
    home_wins_predicted = (results_df['predicted_winner'] == results_df['home_team']).sum()
    home_wins_actual = (results_df['actual_winner'] == results_df['home_team']).sum()
    
    analysis = {
        'overall': {
            'total_games': total_games,
            'correct_predictions': int(correct_predictions),
            'accuracy': overall_accuracy,
            'brier_score': brier,
            'log_loss': logloss,
            'mean_spread_error': mean_spread_error,
            'median_spread_error': median_spread_error,
            'std_spread_error': std_spread_error,
            'rmse_spread': rmse_spread,
        },
        'market_comparison': {
            'market_accuracy': market_accuracy,
            'market_brier': market_brier,
            'model_vs_market_accuracy_diff': overall_accuracy - market_accuracy,
            'model_vs_market_brier_diff': brier - market_brier,
        },
        'roi_results': roi_results,
        'season_metrics': season_metrics,
        'week_metrics': week_metrics,
        'confidence_metrics': confidence_metrics.to_dict('records'),
        'calibration_buckets': cal_buckets_df.to_dict('records'),
        'home_away': {
            'home_wins_predicted': int(home_wins_predicted),
            'home_wins_actual': int(home_wins_actual),
            'home_win_rate_predicted': home_wins_predicted / total_games,
            'home_win_rate_actual': home_wins_actual / total_games,
        },
    }
    
    logger.info("✓ Analysis complete")
    return analysis


def generate_report(
    results_df: pd.DataFrame,
    analysis: Dict,
    output_dir: Path,
) -> str:
    """
    Generate comprehensive markdown report.
    
    Args:
        results_df: Results dataframe
        analysis: Analysis dictionary
        output_dir: Output directory
    
    Returns:
        Path to generated report
    """
    logger.info(f"\n[Step 4/5] Generating report...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"backtest_2019_2022_report.md"
    
    lines = []
    lines.append("# NFL Prediction Model Backtest Report")
    lines.append(f"## Seasons 2023-2024 (Proper Holdout - No Leakage)")
    lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Add leakage verification note
    backtest_config = load_backtest_config()
    train_seasons = backtest_config['splits']['train_seasons']
    val_season = backtest_config['splits']['validation_season']
    lines.append(f"\n**Training Configuration:**")
    lines.append(f"- Training seasons: {train_seasons}")
    lines.append(f"- Validation season: {val_season}")
    lines.append(f"- Test seasons: 2023-2024 (proper holdout, no leakage)")
    lines.append(f"\n---\n")
    
    # Overall Summary
    lines.append("## Overall Summary")
    overall = analysis['overall']
    lines.append(f"\n- **Total Games:** {overall['total_games']}")
    lines.append(f"- **Correct Predictions:** {overall['correct_predictions']}")
    lines.append(f"- **Accuracy:** {overall['accuracy']:.2%}")
    lines.append(f"- **Brier Score:** {overall['brier_score']:.4f}")
    lines.append(f"- **Log Loss:** {overall['log_loss']:.4f}")
    lines.append(f"- **Mean Spread Error:** {overall['mean_spread_error']:.2f} points")
    lines.append(f"- **RMSE Spread:** {overall['rmse_spread']:.2f} points")
    
    # Market Comparison
    lines.append("\n## Market Comparison")
    market = analysis['market_comparison']
    lines.append(f"\n- **Market Accuracy:** {market['market_accuracy']:.2%}")
    lines.append(f"- **Model Accuracy:** {overall['accuracy']:.2%}")
    lines.append(f"- **Difference:** {market['model_vs_market_accuracy_diff']:+.2%}")
    lines.append(f"\n- **Market Brier Score:** {market['market_brier']:.4f}")
    lines.append(f"- **Model Brier Score:** {overall['brier_score']:.4f}")
    lines.append(f"- **Difference:** {market['model_vs_market_brier_diff']:+.4f}")
    
    # ROI Results
    lines.append("\n## ROI vs Market")
    for edge_key, roi_dict in analysis['roi_results'].items():
        edge_threshold = float(edge_key.split('_')[1])
        lines.append(f"\n### Edge Threshold: {edge_threshold:.0%}")
        # Handle different ROI dict structures
        total_bets = roi_dict.get('total_bets', roi_dict.get('n_bets', 0))
        wins = roi_dict.get('wins', roi_dict.get('n_wins', 0))
        losses = roi_dict.get('losses', roi_dict.get('n_losses', 0))
        roi = roi_dict.get('roi', 0.0)
        profit = roi_dict.get('profit', roi_dict.get('total_profit', 0.0))
        lines.append(f"- **Total Bets:** {total_bets}")
        lines.append(f"- **Wins:** {wins}")
        lines.append(f"- **Losses:** {losses}")
        lines.append(f"- **ROI:** {roi:.2%}")
        lines.append(f"- **Profit:** ${profit:.2f}")
    
    # Season-by-Season
    lines.append("\n## Season-by-Season Performance")
    lines.append("\n| Season | Games | Accuracy | Brier Score | Mean Spread Error |")
    lines.append("|--------|-------|----------|-------------|-------------------|")
    for season_metric in analysis['season_metrics']:
        lines.append(
            f"| {season_metric['season']} | {season_metric['n_games']} | "
            f"{season_metric['accuracy']:.2%} | {season_metric['brier_score']:.4f} | "
            f"{season_metric['mean_spread_error']:.2f} |"
        )
    
    # Confidence Analysis
    lines.append("\n## Confidence Analysis")
    lines.append("\n| Confidence | Correct | Total | Accuracy | Mean Spread Error |")
    lines.append("|------------|---------|-------|----------|-------------------|")
    for conf_metric in analysis['confidence_metrics']:
        lines.append(
            f"| {conf_metric['confidence_bin']} | {conf_metric['correct']} | "
            f"{conf_metric['total']} | {conf_metric['accuracy']:.2%} | "
            f"{conf_metric['mean_spread_error']:.2f} |"
        )
    
    # Calibration
    lines.append("\n## Calibration Analysis")
    lines.append("\n| Bin Range | Predicted | Actual | Count | Error |")
    lines.append("|-----------|-----------|--------|-------|-------|")
    for bucket in analysis['calibration_buckets']:
        lines.append(
            f"| [{bucket['bin_min']:.2f}, {bucket['bin_max']:.2f}] | "
            f"{bucket['predicted_freq']:.3f} | {bucket['actual_freq']:.3f} | "
            f"{bucket['count']} | {bucket['calibration_error']:.3f} |"
        )
    
    # Home/Away
    lines.append("\n## Home vs Away Analysis")
    home_away = analysis['home_away']
    lines.append(f"\n- **Home Wins Predicted:** {home_away['home_wins_predicted']} ({home_away['home_win_rate_predicted']:.2%})")
    lines.append(f"- **Home Wins Actual:** {home_away['home_wins_actual']} ({home_away['home_win_rate_actual']:.2%})")
    
    # Write report
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"✓ Report saved to {report_path}")
    return str(report_path)


def verify_no_leakage(seasons: List[int], train_seasons: List[int], val_season: int) -> bool:
    """
    Verify no data leakage between test seasons and training/validation.
    
    Args:
        seasons: Test seasons to verify
        train_seasons: Training seasons
        val_season: Validation season
    
    Returns:
        True if no leakage detected
    """
    train_set = set(train_seasons)
    val_set = {val_season}
    
    leakage_seasons = []
    for season in seasons:
        if season in train_set:
            leakage_seasons.append(f"{season} (in training)")
        elif season in val_set:
            leakage_seasons.append(f"{season} (in validation)")
    
    if leakage_seasons:
        logger.error("=" * 80)
        logger.error("DATA LEAKAGE DETECTED!")
        logger.error("=" * 80)
        logger.error(f"Test seasons with leakage: {leakage_seasons}")
        logger.error(f"Training seasons: {train_seasons}")
        logger.error(f"Validation season: {val_season}")
        logger.error("=" * 80)
        return False
    
    logger.info("✓ No leakage detected - all test seasons are proper holdouts")
    return True


def main():
    """Main function."""
    logger.info("=" * 80)
    logger.info("NFL Prediction Model: Full Backtest Simulation")
    logger.info("Seasons: 2023-2024 (Proper Holdout - No Leakage)")
    logger.info("=" * 80)
    
    # Set up paths
    project_root = Path(__file__).parent.parent
    model_path = project_root / "artifacts" / "models" / "nfl_stacked_ensemble" / "ensemble_v1.pkl"
    output_dir = project_root / "logs" / "simulations" / "backtest_2023_2024"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training config to verify no leakage
    backtest_config = load_backtest_config()
    train_seasons = backtest_config['splits']['train_seasons']
    val_season = backtest_config['splits']['validation_season']
    
    logger.info(f"\nTraining Configuration:")
    logger.info(f"  Training seasons: {train_seasons}")
    logger.info(f"  Validation season: {val_season}")
    
    # Test seasons - proper holdouts
    seasons = [2023, 2024]
    
    # Verify no leakage
    logger.info(f"\n[Step 0/6] Verifying no data leakage...")
    if not verify_no_leakage(seasons, train_seasons, val_season):
        raise ValueError("Data leakage detected! Cannot proceed with backtest.")
    
    # Load model
    logger.info(f"\n[Step 1/6] Loading model...")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = load_model(model_path)
    logger.info(f"✓ Model loaded: {type(model).__name__}")
    
    # Simulate seasons
    logger.info(f"\n[Step 2/6] Simulating predictions for seasons {seasons}...")
    results_df = simulate_seasons(model, seasons)
    
    # Save raw results
    results_path = output_dir / "results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"✓ Saved results to {results_path}")
    
    # Load full data for analysis
    X, y, feature_cols, df = load_features(feature_table="baseline")
    season_mask = df["season"].isin(seasons)
    X_seasons = X[season_mask].copy()
    y_seasons = y[season_mask].copy()
    df_seasons = df[season_mask].copy()
    
    # Analyze results
    analysis = analyze_results(results_df, model, X_seasons, y_seasons, df_seasons)
    
    # Save analysis JSON
    analysis_path = output_dir / "analysis.json"
    analysis_export = {k: v for k, v in analysis.items()}
    with open(analysis_path, 'w') as f:
        json.dump(analysis_export, f, indent=2, default=str)
    logger.info(f"✓ Saved analysis to {analysis_path}")
    
    # Generate report
    logger.info(f"\n[Step 5/6] Generating report...")
    report_path = generate_report(results_df, analysis, output_dir)
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("BACKTEST SIMULATION COMPLETE")
    logger.info("=" * 80)
    overall = analysis['overall']
    logger.info(f"\nTest Seasons: {seasons} (Proper Holdout - No Leakage)")
    logger.info(f"Overall Accuracy: {overall['accuracy']:.2%}")
    logger.info(f"Brier Score: {overall['brier_score']:.4f}")
    logger.info(f"Mean Spread Error: {overall['mean_spread_error']:.2f} points")
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"Report: {report_path}")
    logger.info("=" * 80)
    
    return results_df, analysis


if __name__ == "__main__":
    main()

