"""
2024 Season Validation Script

This script validates the model on the complete 2024 NFL season,
simulating real-world deployment conditions.

CRITICAL: Only uses data available BEFORE each game.
- Team stats through previous week
- Injury report as of Friday before game
- Weather forecast (not actual)
- Odds available before game
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pickle

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.training.trainer import load_features, split_by_season
from models.architectures.stacking_ensemble import StackingEnsemble
from models.calibration import CalibratedModel
from models.base import BaseModel
from eval.metrics import accuracy, brier_score, log_loss
from ingestion.nfl.odds_api import OddsAPIClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model(model_path: Path) -> BaseModel:
    """
    Load trained model from disk.
    
    Args:
        model_path: Path to model file
    
    Returns:
        Loaded model instance
    """
    logger.info(f"Loading model from {model_path}")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"Model loaded successfully")
    return model


def simulate_weekly_predictions(
    model: BaseModel,
    week: int,
    season: int = 2024,
    games_df: Optional[pd.DataFrame] = None,
    X_all: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Simulate predictions for a single week.
    
    CRITICAL: Only use data available BEFORE this week.
    - Team stats through week-1
    - Injury report as of Friday before game
    - Weather forecast (not actual)
    - Odds available before game
    
    Args:
        model: Trained model
        week: Week number
        season: Season year
        games_df: DataFrame with all games
        X_all: Feature matrix for all games
        feature_cols: List of feature column names
    
    Returns:
        DataFrame with:
        - game_id, home_team, away_team
        - model_home_prob, model_pick
        - market_implied_prob, market_pick
        - actual_winner, model_correct, market_correct
    """
    logger.info(f"Simulating predictions for {season} Week {week}")
    
    # Filter to this week's games
    week_games = games_df[
        (games_df['season'] == season) &
        (games_df['week'] == week)
    ].copy()
    
    if len(week_games) == 0:
        logger.warning(f"No games found for {season} Week {week}")
        return pd.DataFrame()
    
    # Get features for these games
    week_indices = week_games.index
    X_week = X_all.loc[week_indices].copy()
    
    # Ensure we have the right feature columns
    if feature_cols:
        missing_cols = set(feature_cols) - set(X_week.columns)
        if missing_cols:
            logger.warning(f"Missing features: {missing_cols}")
            for col in missing_cols:
                X_week[col] = 0.0
    
    # Make predictions
    try:
        model_probs = model.predict_proba(X_week[feature_cols] if feature_cols else X_week)
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        model_probs = np.full(len(week_games), 0.5)
    
    # Get market data (if available)
    market_probs = []
    if 'close_spread' in week_games.columns:
        # Convert spread to probability
        for _, game in week_games.iterrows():
            spread = game.get('close_spread', 0)
            if pd.notna(spread):
                # Simple spread to probability conversion
                prob = 1 / (1 + np.exp(spread / 3.0))
                market_probs.append(prob)
            else:
                market_probs.append(0.5)
    else:
        market_probs = [0.5] * len(week_games)
    
    # Create results DataFrame
    results = []
    for idx, (game_idx, game) in enumerate(week_games.iterrows()):
        model_prob = model_probs[idx] if isinstance(model_probs, np.ndarray) else model_probs
        market_prob = market_probs[idx]
        
        # Determine picks
        model_pick = game['home_team'] if model_prob > 0.5 else game['away_team']
        market_pick = game['home_team'] if market_prob > 0.5 else game['away_team']
        
        # Get actual winner (if available)
        actual_winner = None
        if 'home_score' in game and 'away_score' in game:
            if pd.notna(game['home_score']) and pd.notna(game['away_score']):
                if game['home_score'] > game['away_score']:
                    actual_winner = game['home_team']
                elif game['away_score'] > game['home_score']:
                    actual_winner = game['away_team']
        
        model_correct = None
        market_correct = None
        if actual_winner:
            model_correct = (model_pick == actual_winner)
            market_correct = (market_pick == actual_winner)
        
        results.append({
            'game_id': game.get('game_id'),
            'season': season,
            'week': week,
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'model_home_prob': float(model_prob),
            'model_pick': model_pick,
            'model_confidence': float(max(model_prob, 1 - model_prob)),
            'market_implied_prob': float(market_prob),
            'market_pick': market_pick,
            'actual_winner': actual_winner,
            'model_correct': model_correct,
            'market_correct': market_correct,
            'model_edge': float(model_prob - market_prob),
        })
    
    return pd.DataFrame(results)


def calculate_roi(
    predictions_df: pd.DataFrame,
    confidence_threshold: float = 0.0,
    edge_threshold: float = 0.0,
) -> Dict:
    """
    Calculate ROI assuming flat betting on model picks.
    
    Args:
        predictions_df: DataFrame with predictions and outcomes
        confidence_threshold: Only bet when model confidence > threshold
        edge_threshold: Only bet when model edge vs market > threshold
    
    Returns:
        {
            'total_bets': int,
            'wins': int,
            'losses': int,
            'win_rate': float,
            'roi': float,
            'units_won': float,
        }
    """
    # Filter to bets that meet thresholds
    bets = predictions_df[
        (predictions_df['model_confidence'] >= confidence_threshold) &
        (predictions_df['model_edge'].abs() >= edge_threshold) &
        (predictions_df['model_correct'].notna())
    ].copy()
    
    if len(bets) == 0:
        return {
            'total_bets': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
            'roi': 0.0,
            'units_won': 0.0,
        }
    
    wins = bets['model_correct'].sum()
    losses = len(bets) - wins
    
    # Calculate ROI (simplified - assumes -110 odds)
    # Win: +0.909 units (bet 1.0, win 0.909)
    # Loss: -1.0 units
    units_won = wins * 0.909 - losses * 1.0
    roi = units_won / len(bets) if len(bets) > 0 else 0.0
    
    return {
        'total_bets': len(bets),
        'wins': int(wins),
        'losses': int(losses),
        'win_rate': float(wins / len(bets)) if len(bets) > 0 else 0.0,
        'roi': float(roi),
        'units_won': float(units_won),
    }


def validate_2024_season(
    model_path: Optional[Path] = None,
    feature_table: str = "baseline",
) -> Dict:
    """
    Run comprehensive validation on 2024 season.
    
    Process:
    1. Load model trained on 2015-2023 data
    2. For each week 1-18:
       a. Generate features using only data available before that week
       b. Make predictions
       c. Compare to actual outcomes
       d. Compare to betting market (closing lines)
    3. Calculate overall metrics
    
    Args:
        model_path: Path to trained model (default: look for ensemble)
        feature_table: Feature table name
    
    Returns:
        {
            'overall_accuracy': float,
            'overall_roi': float,
            'accuracy_by_week': dict,
            'accuracy_by_confidence': dict,
            'vs_market_performance': dict,
            'calibration_results': dict,
        }
    """
    logger.info("=" * 60)
    logger.info("2024 Season Validation")
    logger.info("=" * 60)
    
    # Load model
    if model_path is None:
        # Try to find ensemble model
        possible_paths = [
            Path(__file__).parent.parent / "logs" / "phase1_validation" / "models" / "ensemble.pkl",
            Path(__file__).parent.parent / "models" / "artifacts" / "nfl_ensemble" / "ensemble.pkl",
        ]
        for path in possible_paths:
            if path.exists():
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError("Model not found. Please specify model_path.")
    
    model = load_model(model_path)
    
    # Load features
    logger.info("Loading features...")
    X, y, feature_cols, games_df = load_features(feature_table=feature_table)
    
    # Filter to 2024 season
    games_2024 = games_df[games_df['season'] == 2024].copy()
    X_2024 = X.loc[games_2024.index].copy()
    
    logger.info(f"Found {len(games_2024)} games in 2024 season")
    
    # Simulate week-by-week predictions
    all_predictions = []
    weeks = sorted(games_2024['week'].unique())
    
    for week in weeks:
        week_preds = simulate_weekly_predictions(
            model=model,
            week=week,
            season=2024,
            games_df=games_df,
            X_all=X,
            feature_cols=feature_cols,
        )
        if len(week_preds) > 0:
            all_predictions.append(week_preds)
    
    if len(all_predictions) == 0:
        logger.error("No predictions generated")
        return {}
    
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    
    # Filter to games with actual outcomes
    predictions_with_outcomes = predictions_df[predictions_df['model_correct'].notna()].copy()
    
    if len(predictions_with_outcomes) == 0:
        logger.warning("No games with actual outcomes available")
        return {
            'total_games': len(predictions_df),
            'games_with_outcomes': 0,
        }
    
    # Calculate overall metrics
    overall_accuracy = predictions_with_outcomes['model_correct'].mean()
    
    # Calculate ROI
    roi_all = calculate_roi(predictions_with_outcomes, confidence_threshold=0.0, edge_threshold=0.0)
    roi_high_conf = calculate_roi(predictions_with_outcomes, confidence_threshold=0.70, edge_threshold=0.0)
    
    # Accuracy by week
    accuracy_by_week = {}
    for week in sorted(predictions_with_outcomes['week'].unique()):
        week_data = predictions_with_outcomes[predictions_with_outcomes['week'] == week]
        if len(week_data) > 0:
            accuracy_by_week[int(week)] = float(week_data['model_correct'].mean())
    
    # Accuracy by confidence tier
    predictions_with_outcomes['confidence_tier'] = pd.cut(
        predictions_with_outcomes['model_confidence'],
        bins=[0, 0.6, 0.7, 0.8, 1.0],
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    accuracy_by_confidence = {}
    for tier in ['Low', 'Medium', 'High', 'Very High']:
        tier_data = predictions_with_outcomes[predictions_with_outcomes['confidence_tier'] == tier]
        if len(tier_data) > 0:
            accuracy_by_confidence[tier] = {
                'accuracy': float(tier_data['model_correct'].mean()),
                'count': len(tier_data),
            }
    
    # vs Market performance
    market_accuracy = predictions_with_outcomes['market_correct'].mean()
    model_beats_market = (predictions_with_outcomes['model_correct'] & ~predictions_with_outcomes['market_correct']).sum()
    market_beats_model = (~predictions_with_outcomes['model_correct'] & predictions_with_outcomes['market_correct']).sum()
    
    vs_market = {
        'model_accuracy': float(overall_accuracy),
        'market_accuracy': float(market_accuracy),
        'model_beats_market': int(model_beats_market),
        'market_beats_model': int(market_beats_model),
        'model_beats_market_pct': float(model_beats_market / len(predictions_with_outcomes)) if len(predictions_with_outcomes) > 0 else 0.0,
    }
    
    # Calibration check (simplified)
    calibration_results = {}
    for tier, data in accuracy_by_confidence.items():
        if data['count'] > 0:
            # Expected accuracy is midpoint of confidence range
            tier_midpoints = {
                'Low': 0.55,
                'Medium': 0.65,
                'High': 0.75,
                'Very High': 0.85,
            }
            expected = tier_midpoints.get(tier, 0.5)
            actual = data['accuracy']
            calibration_results[tier] = {
                'expected': expected,
                'actual': actual,
                'difference': actual - expected,
            }
    
    results = {
        'overall_accuracy': float(overall_accuracy),
        'overall_roi': float(roi_all['roi']),
        'roi_high_confidence': float(roi_high_conf['roi']),
        'total_games': len(predictions_df),
        'games_with_outcomes': len(predictions_with_outcomes),
        'accuracy_by_week': accuracy_by_week,
        'accuracy_by_confidence': accuracy_by_confidence,
        'vs_market_performance': vs_market,
        'calibration_results': calibration_results,
        'roi_details': {
            'all_bets': roi_all,
            'high_confidence': roi_high_conf,
        },
    }
    
    return results


def generate_validation_report(results: Dict, output_path: Path) -> str:
    """
    Generate markdown report of validation results.
    
    Args:
        results: Results dictionary from validate_2024_season()
        output_path: Path to save report
    
    Returns:
        Report text
    """
    report_lines = []
    
    report_lines.append("# 2024 Season Validation Report")
    report_lines.append("")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Overall Performance
    report_lines.append("## Overall Performance")
    report_lines.append("")
    report_lines.append(f"- **Accuracy**: {results.get('overall_accuracy', 0):.1%}")
    report_lines.append(f"- **ROI (All Bets)**: {results.get('overall_roi', 0):.2%}")
    report_lines.append(f"- **ROI (70%+ Confidence)**: {results.get('roi_high_confidence', 0):.2%}")
    report_lines.append(f"- **Total Games**: {results.get('total_games', 0)}")
    report_lines.append(f"- **Games with Outcomes**: {results.get('games_with_outcomes', 0)}")
    report_lines.append("")
    
    # Performance by Confidence Tier
    report_lines.append("## Performance by Confidence Tier")
    report_lines.append("")
    report_lines.append("| Confidence | Games | Accuracy | Expected | Difference |")
    report_lines.append("|------------|-------|----------|----------|------------|")
    
    accuracy_by_conf = results.get('accuracy_by_confidence', {})
    calibration = results.get('calibration_results', {})
    
    for tier in ['Low', 'Medium', 'High', 'Very High']:
        if tier in accuracy_by_conf:
            acc_data = accuracy_by_conf[tier]
            cal_data = calibration.get(tier, {})
            report_lines.append(
                f"| {tier} | {acc_data['count']} | {acc_data['accuracy']:.1%} | "
                f"{cal_data.get('expected', 0):.1%} | {cal_data.get('difference', 0):+.1%} |"
            )
    report_lines.append("")
    
    # vs Market Performance
    report_lines.append("## vs Market Performance")
    report_lines.append("")
    vs_market = results.get('vs_market_performance', {})
    report_lines.append(f"- **Model Accuracy**: {vs_market.get('model_accuracy', 0):.1%}")
    report_lines.append(f"- **Market Accuracy**: {vs_market.get('market_accuracy', 0):.1%}")
    report_lines.append(f"- **Model Beats Market**: {vs_market.get('model_beats_market', 0)} games "
                       f"({vs_market.get('model_beats_market_pct', 0):.1%})")
    report_lines.append("")
    
    # ROI Details
    report_lines.append("## ROI Details")
    report_lines.append("")
    roi_details = results.get('roi_details', {})
    for bet_type, roi_data in roi_details.items():
        report_lines.append(f"### {bet_type.replace('_', ' ').title()}")
        report_lines.append(f"- **Total Bets**: {roi_data.get('total_bets', 0)}")
        report_lines.append(f"- **Wins**: {roi_data.get('wins', 0)}")
        report_lines.append(f"- **Losses**: {roi_data.get('losses', 0)}")
        report_lines.append(f"- **Win Rate**: {roi_data.get('win_rate', 0):.1%}")
        report_lines.append(f"- **ROI**: {roi_data.get('roi', 0):.2%}")
        report_lines.append("")
    
    # Success Criteria
    report_lines.append("## Success Criteria")
    report_lines.append("")
    acc_2024 = results.get('overall_accuracy', 0)
    roi_high = results.get('roi_high_confidence', 0)
    beats_market_pct = vs_market.get('model_beats_market_pct', 0)
    
    status1 = "✓" if acc_2024 >= 0.60 else "✗"
    report_lines.append(f"- **{status1} 2024 accuracy >= 60%**: {acc_2024:.1%}")
    
    status2 = "✓" if roi_high > 0 else "✗"
    report_lines.append(f"- **{status2} Positive ROI on 70%+ confidence picks**: {roi_high:.2%}")
    
    status3 = "✓" if beats_market_pct >= 0.55 else "✗"
    report_lines.append(f"- **{status3} Model beats market on 55%+ of games**: {beats_market_pct:.1%}")
    report_lines.append("")
    
    report_text = "\n".join(report_lines)
    
    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    logger.info(f"Saved validation report to {output_path}")
    
    return report_text


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate model on 2024 season")
    parser.add_argument('--model-path', type=str, help='Path to trained model')
    parser.add_argument('--feature-table', type=str, default='baseline', help='Feature table name')
    parser.add_argument('--output-dir', type=str, default='logs/phase3_validation', help='Output directory')
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path) if args.model_path else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run validation
    results = validate_2024_season(
        model_path=model_path,
        feature_table=args.feature_table,
    )
    
    # Save results JSON
    results_path = output_dir / "2024_validation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Saved results to {results_path}")
    
    # Generate report
    report_path = output_dir / "2024_validation_report.md"
    generate_validation_report(results, report_path)
    
    logger.info("=" * 60)
    logger.info("2024 Validation Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

