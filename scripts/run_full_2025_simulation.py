"""
Run full 2025 season simulation - predict all games and compare with actual results.

Generates comprehensive analysis of prediction performance.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Dict, List
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.simulate_real_world_prediction import (
    load_ensemble_model,
    get_game_features,
    prepare_features_for_prediction,
    get_actual_results,
    predict_spread,
    calculate_spread,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_full_season_simulation(
    season: int = 2025,
    model_path: Path = None,
    feature_table: str = "baseline",
) -> pd.DataFrame:
    """
    Run predictions for all games in a season.
    
    Args:
        season: Season year
        model_path: Path to ensemble model
        feature_table: Feature table name
    
    Returns:
        DataFrame with predictions and actual results
    """
    logger.info("=" * 80)
    logger.info(f"Full Season Simulation: {season}")
    logger.info("=" * 80)
    
    # Load model
    if model_path is None:
        model_path = Path(__file__).parent.parent / "artifacts" / "models" / "nfl_stacked_ensemble_v2" / "ensemble_v1.pkl"
        if not model_path.exists():
            model_path = Path(__file__).parent.parent / "artifacts" / "models" / "nfl_stacked_ensemble" / "ensemble_v1.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    logger.info(f"Loading ensemble model from {model_path}")
    model = load_ensemble_model(model_path)
    
    # Load features
    from features.feature_table_registry import get_feature_table_path
    features_path = get_feature_table_path(feature_table)
    logger.info(f"Loading features from {features_path}")
    features_df = pd.read_parquet(features_path)
    
    # Filter to season
    season_games = features_df[features_df['season'] == season].copy()
    logger.info(f"Found {len(season_games)} games for {season}")
    
    # Load games for actual results
    games_path = Path(__file__).parent.parent / "data" / "nfl" / "staged" / "games.parquet"
    games_df = pd.read_parquet(games_path)
    
    results = []
    
    for idx, row in season_games.iterrows():
        game_id = row['game_id']
        home_team = row['home_team']
        away_team = row['away_team']
        
        try:
            # Prepare features
            game_data = pd.DataFrame([row])
            X, feature_cols = prepare_features_for_prediction(game_data)
            
            # Run prediction
            predicted_prob = model.predict_proba(X)[0]
            predicted_winner = home_team if predicted_prob >= 0.5 else away_team
            predicted_spread = predict_spread(predicted_prob)
            confidence = max(predicted_prob, 1 - predicted_prob)
            
            # Get actual results
            actual_results = get_actual_results(game_id)
            
            if actual_results and actual_results.get('home_score') is not None and actual_results.get('away_score') is not None:
                actual_home_score = int(actual_results['home_score'])
                actual_away_score = int(actual_results['away_score'])
                actual_winner = home_team if actual_home_score > actual_away_score else away_team
                actual_spread = calculate_spread(actual_home_score, actual_away_score)
                correct = (predicted_winner == actual_winner)
                
                results.append({
                    'game_id': game_id,
                    'season': season,
                    'week': row['week'],
                    'home_team': home_team,
                    'away_team': away_team,
                    'predicted_winner': predicted_winner,
                    'predicted_prob': predicted_prob,
                    'predicted_spread': predicted_spread,
                    'confidence': confidence,
                    'actual_winner': actual_winner,
                    'actual_home_score': actual_home_score,
                    'actual_away_score': actual_away_score,
                    'actual_spread': actual_spread,
                    'correct': correct,
                    'spread_error': abs(predicted_spread - actual_spread),
                    'predicted_margin': abs(predicted_spread),
                    'actual_margin': abs(actual_spread),
                })
            else:
                # Game not played yet or no scores available
                results.append({
                    'game_id': game_id,
                    'season': season,
                    'week': row['week'],
                    'home_team': home_team,
                    'away_team': away_team,
                    'predicted_winner': predicted_winner,
                    'predicted_prob': predicted_prob,
                    'predicted_spread': predicted_spread,
                    'confidence': confidence,
                    'actual_winner': None,
                    'actual_home_score': None,
                    'actual_away_score': None,
                    'actual_spread': None,
                    'correct': None,
                    'spread_error': None,
                    'predicted_margin': abs(predicted_spread),
                    'actual_margin': None,
                })
            
            if (idx + 1) % 50 == 0:
                logger.info(f"Processed {idx + 1}/{len(season_games)} games...")
                
        except Exception as e:
            logger.error(f"Error processing {game_id}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    logger.info(f"Completed predictions for {len(results_df)} games")
    
    return results_df


def analyze_results(results_df: pd.DataFrame) -> Dict:
    """
    Analyze prediction results and calculate metrics.
    
    Args:
        results_df: DataFrame with predictions and actuals
    
    Returns:
        Dictionary with analysis metrics
    """
    # Filter to games with actual results
    completed_games = results_df[results_df['correct'].notna()].copy()
    
    if len(completed_games) == 0:
        return {"error": "No completed games found"}
    
    total_games = len(completed_games)
    correct_predictions = completed_games['correct'].sum()
    accuracy = correct_predictions / total_games
    
    # Spread error analysis
    spread_errors = completed_games['spread_error'].dropna()
    mean_spread_error = spread_errors.mean()
    median_spread_error = spread_errors.median()
    std_spread_error = spread_errors.std()
    
    # Confidence analysis
    correct_high_conf = completed_games[
        (completed_games['correct'] == True) & (completed_games['confidence'] >= 0.6)
    ]
    incorrect_high_conf = completed_games[
        (completed_games['correct'] == False) & (completed_games['confidence'] >= 0.6)
    ]
    
    # Home vs away predictions
    home_wins_predicted = (completed_games['predicted_winner'] == completed_games['home_team']).sum()
    home_wins_actual = (completed_games['actual_winner'] == completed_games['home_team']).sum()
    
    # Week-by-week analysis
    week_accuracy = completed_games.groupby('week').agg({
        'correct': ['sum', 'count', lambda x: x.sum() / len(x)]
    }).reset_index()
    week_accuracy.columns = ['week', 'correct', 'total', 'accuracy']
    
    # Confidence bins
    completed_games['confidence_bin'] = pd.cut(
        completed_games['confidence'],
        bins=[0, 0.5, 0.6, 0.7, 0.8, 1.0],
        labels=['<50%', '50-60%', '60-70%', '70-80%', '80%+']
    )
    confidence_accuracy = completed_games.groupby('confidence_bin').agg({
        'correct': ['sum', 'count', lambda x: x.sum() / len(x)]
    }).reset_index()
    confidence_accuracy.columns = ['confidence_bin', 'correct', 'total', 'accuracy']
    
    return {
        'total_games': total_games,
        'correct_predictions': int(correct_predictions),
        'accuracy': accuracy,
        'mean_spread_error': mean_spread_error,
        'median_spread_error': median_spread_error,
        'std_spread_error': std_spread_error,
        'high_confidence_correct': len(correct_high_conf),
        'high_confidence_incorrect': len(incorrect_high_conf),
        'home_wins_predicted': int(home_wins_predicted),
        'home_wins_actual': int(home_wins_actual),
        'home_win_rate_predicted': home_wins_predicted / total_games,
        'home_win_rate_actual': home_wins_actual / total_games,
        'week_accuracy': week_accuracy,
        'confidence_accuracy': confidence_accuracy,
        'results_df': completed_games,
    }


def main():
    """Main function."""
    logger.info("Starting full 2025 season simulation...")
    
    # Run predictions
    results_df = run_full_season_simulation(season=2025)
    
    # Save results
    output_dir = Path(__file__).parent.parent / "logs" / "simulations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / f"full_2025_simulation_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved results to {results_path}")
    
    # Analyze results
    logger.info("\nAnalyzing results...")
    analysis = analyze_results(results_df)
    
    if 'error' in analysis:
        logger.error(analysis['error'])
        return
    
    # Print summary
    print("\n" + "=" * 80)
    print("2025 SEASON SIMULATION RESULTS")
    print("=" * 80)
    print(f"\nTotal Games: {analysis['total_games']}")
    print(f"Correct Predictions: {analysis['correct_predictions']}")
    print(f"Accuracy: {analysis['accuracy']:.2%}")
    print(f"\nSpread Error:")
    print(f"  Mean: {analysis['mean_spread_error']:.2f} points")
    print(f"  Median: {analysis['median_spread_error']:.2f} points")
    print(f"  Std Dev: {analysis['std_spread_error']:.2f} points")
    print(f"\nHome Win Rate:")
    print(f"  Predicted: {analysis['home_win_rate_predicted']:.2%}")
    print(f"  Actual: {analysis['home_win_rate_actual']:.2%}")
    print(f"\nHigh Confidence Predictions (≥60%):")
    print(f"  Correct: {analysis['high_confidence_correct']}")
    print(f"  Incorrect: {analysis['high_confidence_incorrect']}")
    
    # Save analysis
    analysis_path = output_dir / f"full_2025_simulation_analysis.json"
    import json
    analysis_export = {k: v for k, v in analysis.items() if k != 'results_df'}
    analysis_export['week_accuracy'] = analysis['week_accuracy'].to_dict('records')
    analysis_export['confidence_accuracy'] = analysis['confidence_accuracy'].to_dict('records')
    with open(analysis_path, 'w') as f:
        json.dump(analysis_export, f, indent=2, default=str)
    logger.info(f"Saved analysis to {analysis_path}")
    
    return results_df, analysis


if __name__ == "__main__":
    main()

