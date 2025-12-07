"""
Simulate real-world prediction for a specific NFL game.

Uses only pre-game data (odds, injuries, schedule, team stats, roster as of game day).
Loads ensemble model and runs inference using production pipeline.

Usage:
    python scripts/simulate_real_world_prediction.py --game-id 2025_WK14_DAL_DET
    python scripts/simulate_real_world_prediction.py  # Uses default: 2025_WK14_DAL_DET
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Optional, Dict, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.architectures.stacking_ensemble import StackingEnsemble
from models.architectures.ft_transformer import FTTransformerModel
from models.architectures.tabnet import TabNetModel
from models.architectures.gradient_boosting import GradientBoostingModel
from models.base import BaseModel
from features.feature_table_registry import get_feature_table_path, validate_feature_table_exists
from ingestion.nfl.schedule import form_game_id, normalize_team_abbreviation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_game_id(game_id_str: str) -> Tuple[int, int, str, str]:
    """
    Parse game ID from user format to components.
    
    Formats supported:
    - 2025_WK14_DAL_DET (user format: HOME_AWAY, e.g., Dallas home vs Detroit away)
    - nfl_2025_14_DET_DAL (our format: AWAY_HOME)
    
    Args:
        game_id_str: Game ID string
    
    Returns:
        Tuple of (season, week, away_team, home_team)
    """
    # Try user format first: 2025_WK14_DAL_DET
    # User format convention: SEASON_WK##_HOME_AWAY
    # Example: 2025_WK14_DAL_DET means Dallas (home) vs Detroit (away)
    if "_WK" in game_id_str.upper() or (game_id_str.count("_") >= 3 and not game_id_str.startswith("nfl_")):
        parts = game_id_str.upper().split("_")
        if len(parts) >= 4:
            season = int(parts[0])
            # Handle WK## or just ##
            week_str = parts[1].replace("WK", "").replace("WEEK", "")
            week = int(week_str)
            # User format: HOME_AWAY (e.g., DAL_DET = Dallas home, Detroit away)
            home_team = normalize_team_abbreviation(parts[2])
            away_team = normalize_team_abbreviation(parts[3])
            return season, week, away_team, home_team
    
    # Try our format: nfl_2025_14_DET_DAL
    if game_id_str.startswith("nfl_"):
        parts = game_id_str.split("_")
        if len(parts) >= 5:
            season = int(parts[1])
            week = int(parts[2])
            away = normalize_team_abbreviation(parts[3])
            home = normalize_team_abbreviation(parts[4])
            return season, week, away, home
    
    raise ValueError(f"Could not parse game_id: {game_id_str}. Expected format: 2025_WK14_DAL_DET or nfl_2025_14_DET_DAL")


def normalize_game_id(season: int, week: int, away_team: str, home_team: str) -> str:
    """Convert to our standard game_id format."""
    return form_game_id(season, week, away_team, home_team)


def load_ensemble_model(model_path: Path) -> BaseModel:
    """Load ensemble model with proper base model loaders."""
    def custom_base_model_loader(path):
        """Load base model with proper type detection."""
        path = Path(path)
        if 'ft_transformer' in str(path):
            return FTTransformerModel.load(path)
        elif 'tabnet' in str(path):
            model = TabNetModel.load(path)
            # Force CPU if CUDA not compatible
            if hasattr(model, 'device') and model.device == 'cuda':
                import torch
                try:
                    test_tensor = torch.zeros(1).cuda()
                    _ = test_tensor + 1
                    del test_tensor
                    torch.cuda.empty_cache()
                except Exception:
                    model.device = 'cpu'
                    if hasattr(model, 'model') and model.model is not None:
                        if hasattr(model.model, 'device'):
                            model.model.device = 'cpu'
                        if hasattr(model.model, 'to'):
                            model.model = model.model.to('cpu')
            return model
        elif 'gbm' in str(path):
            return GradientBoostingModel.load(path)
        else:
            return BaseModel.load(path)
    
    return StackingEnsemble.load(model_path, base_model_loader=custom_base_model_loader)


def find_matching_game(season: int, week: int, team1: str, team2: str, games_df: pd.DataFrame) -> Optional[str]:
    """
    Find a matching game_id when the exact game_id is not found.
    
    Tries to find a game with the same teams (regardless of home/away order).
    
    Args:
        season: Season year
        week: Week number
        team1: First team
        team2: Second team
        games_df: DataFrame with games
    
    Returns:
        Matching game_id or None
    """
    # Find games with these teams in this week
    week_games = games_df[(games_df['season'] == season) & (games_df['week'] == week)]
    
    # Check if either team combination matches
    matches = week_games[
        ((week_games['home_team'] == team1) & (week_games['away_team'] == team2)) |
        ((week_games['home_team'] == team2) & (week_games['away_team'] == team1))
    ]
    
    if len(matches) > 0:
        return matches.iloc[0]['game_id']
    
    return None


def get_game_features(game_id: str, feature_table: str = "baseline") -> Optional[pd.DataFrame]:
    """
    Get features for a specific game from the feature table.
    
    Args:
        game_id: Game ID in format nfl_{season}_{week}_{away}_{home}
        feature_table: Feature table name ("baseline", "phase2", "phase2b")
    
    Returns:
        DataFrame with single row for the game, or None if not found
    """
    validate_feature_table_exists(feature_table)
    features_path = get_feature_table_path(feature_table)
    
    logger.info(f"Loading features from {features_path}")
    df = pd.read_parquet(features_path)
    
    # Find the game
    game_data = df[df['game_id'] == game_id].copy()
    
    if len(game_data) == 0:
        logger.warning(f"Game {game_id} not found in feature table")
        
        # Try to find a matching game with same teams
        # Parse game_id to get teams
        parts = game_id.split('_')
        if len(parts) >= 5:
            season = int(parts[1])
            week = int(parts[2])
            away_team = parts[3]
            home_team = parts[4]
            
            # Load games to find matching game
            games_path = Path(__file__).parent.parent / "data" / "nfl" / "staged" / "games.parquet"
            if games_path.exists():
                games_df = pd.read_parquet(games_path)
                matching_game_id = find_matching_game(season, week, away_team, home_team, games_df)
                
                if matching_game_id:
                    logger.info(f"Found matching game: {matching_game_id} (teams may be reversed)")
                    game_data = df[df['game_id'] == matching_game_id].copy()
                    if len(game_data) > 0:
                        logger.info(f"Using matching game for features")
                        return game_data.iloc[[0]] if len(game_data) > 0 else None
        
        return None
    
    if len(game_data) > 1:
        logger.warning(f"Multiple rows found for {game_id}, using first")
        game_data = game_data.iloc[[0]]
    
    return game_data


def prepare_features_for_prediction(game_data: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    Prepare features for model prediction.
    
    Args:
        game_data: DataFrame with game data (single row)
    
    Returns:
        Tuple of (feature_matrix, feature_columns)
    """
    # Exclude non-feature columns (same as in trainer.py)
    exclude_cols = [
        "game_id",
        "season",
        "week",
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "home_win",
        "close_spread",
        "close_total",
        "open_spread",
        "open_total",
    ]
    
    feature_cols = [col for col in game_data.columns if col not in exclude_cols]
    
    # Extract features
    X = game_data[feature_cols].copy()
    
    # Fill any missing values with 0
    X = X.fillna(0)
    
    return X, feature_cols


def get_actual_results(game_id: str) -> Optional[Dict]:
    """
    Get actual game results from games.parquet.
    
    Args:
        game_id: Game ID
    
    Returns:
        Dictionary with actual results or None if not found
    """
    games_path = Path(__file__).parent.parent / "data" / "nfl" / "staged" / "games.parquet"
    
    if not games_path.exists():
        logger.warning(f"Games file not found: {games_path}")
        return None
    
    df = pd.read_parquet(games_path)
    game_data = df[df['game_id'] == game_id]
    
    if len(game_data) == 0:
        logger.warning(f"Game {game_id} not found in games.parquet")
        return None
    
    row = game_data.iloc[0]
    
    return {
        'home_score': int(row['home_score']) if pd.notna(row['home_score']) else None,
        'away_score': int(row['away_score']) if pd.notna(row['away_score']) else None,
        'home_team': row['home_team'],
        'away_team': row['away_team'],
        'date': row.get('date', None),
    }


def calculate_spread(home_score: int, away_score: int) -> float:
    """Calculate point spread from scores (home perspective)."""
    return home_score - away_score


def predict_spread(predicted_prob: float) -> float:
    """
    Estimate point spread from predicted probability.
    
    Simple approximation: spread ≈ -3 * logit(prob)
    This is a rough estimate - actual spread prediction would require a separate model.
    """
    epsilon = 1e-15
    prob_clipped = np.clip(predicted_prob, epsilon, 1 - epsilon)
    logit = np.log(prob_clipped / (1 - prob_clipped))
    estimated_spread = -3 * logit  # Rough approximation
    return estimated_spread


def log_prediction_result(
    log_path: Path,
    game_id: str,
    predicted_winner: str,
    predicted_prob: float,
    predicted_spread: float,
    actual_winner: Optional[str],
    actual_home_score: Optional[int],
    actual_away_score: Optional[int],
    actual_spread: Optional[float],
    correct: Optional[bool],
    home_team: str,
    away_team: str,
):
    """Log prediction result to file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().isoformat()
    
    log_entry = {
        'timestamp': timestamp,
        'game_id': game_id,
        'home_team': home_team,
        'away_team': away_team,
        'predicted_winner': predicted_winner,
        'predicted_prob': predicted_prob,
        'predicted_spread': predicted_spread,
        'actual_winner': actual_winner if actual_winner else 'N/A',
        'actual_home_score': actual_home_score if actual_home_score is not None else 'N/A',
        'actual_away_score': actual_away_score if actual_away_score is not None else 'N/A',
        'actual_spread': actual_spread if actual_spread is not None else 'N/A',
        'correct': correct if correct is not None else 'N/A',
    }
    
    # Append to log file (CSV format for easy analysis)
    log_df = pd.DataFrame([log_entry])
    
    if log_path.exists():
        existing_df = pd.read_csv(log_path)
        log_df = pd.concat([existing_df, log_df], ignore_index=True)
    
    log_df.to_csv(log_path, index=False)
    logger.info(f"Logged prediction result to {log_path}")


def simulate_prediction(
    game_id: str,
    model_path: Optional[Path] = None,
    feature_table: str = "baseline",
    log_path: Optional[Path] = None,
) -> Dict:
    """
    Simulate prediction for a specific game.
    
    Args:
        game_id: Game ID (will be normalized to our format)
        model_path: Path to ensemble model (default: v2 ensemble)
        feature_table: Feature table name
        log_path: Path to log file
    
    Returns:
        Dictionary with prediction results
    """
    # Parse and normalize game ID
    try:
        season, week, away_team, home_team = parse_game_id(game_id)
        normalized_game_id = normalize_game_id(season, week, away_team, home_team)
        logger.info(f"Parsed game_id: {game_id} -> {normalized_game_id}")
        logger.info(f"  Season: {season}, Week: {week}")
        logger.info(f"  Away: {away_team}, Home: {home_team}")
    except Exception as e:
        logger.error(f"Error parsing game_id: {e}")
        raise
    
    # Load model
    if model_path is None:
        # Default to v2 ensemble (simpler, equivalent performance)
        model_path = Path(__file__).parent.parent / "artifacts" / "models" / "nfl_stacked_ensemble_v2" / "ensemble_v1.pkl"
        # Fallback to v1 if v2 doesn't exist
        if not model_path.exists():
            model_path = Path(__file__).parent.parent / "artifacts" / "models" / "nfl_stacked_ensemble" / "ensemble_v1.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    logger.info(f"Loading ensemble model from {model_path}")
    model = load_ensemble_model(model_path)
    
    # Get game features
    game_data = get_game_features(normalized_game_id, feature_table)
    if game_data is None:
        # Check if game exists in games.parquet (might be scheduled but features not generated)
        games_path = Path(__file__).parent.parent / "data" / "nfl" / "staged" / "games.parquet"
        if games_path.exists():
            games_df = pd.read_parquet(games_path)
            
            # Try to find matching game with same teams
            matching_game_id = find_matching_game(season, week, away_team, home_team, games_df)
            
            if matching_game_id and matching_game_id != normalized_game_id:
                logger.warning(f"Game {normalized_game_id} not found, but found matching game: {matching_game_id}")
                logger.info(f"  This might be due to home/away team order difference")
                logger.info(f"  Trying to use matching game...")
                
                # Try to get features for matching game
                game_data = get_game_features(matching_game_id, feature_table)
                if game_data is not None:
                    logger.info(f"✓ Using matching game {matching_game_id} for prediction")
                    normalized_game_id = matching_game_id  # Update to use matching game_id
                    # Update home/away teams from actual game
                    home_team = game_data.iloc[0]['home_team']
                    away_team = game_data.iloc[0]['away_team']
            
            if game_data is None:
                if normalized_game_id in games_df['game_id'].values or (matching_game_id and matching_game_id in games_df['game_id'].values):
                    raise ValueError(
                        f"Game {normalized_game_id} exists in schedule but features not found. "
                        f"Please generate features for this game first using the feature pipeline."
                    )
        
        if game_data is None:
            raise ValueError(
                f"Game {normalized_game_id} not found in feature table. "
                f"Make sure:\n"
                f"  1. The game exists in the schedule\n"
                f"  2. Features have been generated for this game\n"
                f"  3. The game_id format is correct\n"
                f"\n"
                f"To check available games, query the feature table:\n"
                f"  python3 -c \"import pandas as pd; df = pd.read_parquet('data/nfl/processed/game_features_baseline.parquet'); print(df[df['season']=={season}][df['week']=={week}][['game_id', 'home_team', 'away_team']])\""
            )
    
    # Prepare features for prediction
    X, feature_cols = prepare_features_for_prediction(game_data)
    logger.info(f"Prepared {len(feature_cols)} features for prediction")
    
    # Run prediction
    logger.info("Running model prediction...")
    predicted_prob = model.predict_proba(X)[0]
    predicted_winner = home_team if predicted_prob >= 0.5 else away_team
    predicted_spread = predict_spread(predicted_prob)
    confidence = max(predicted_prob, 1 - predicted_prob)
    
    logger.info(f"Prediction: {predicted_winner} wins (probability: {predicted_prob:.4f}, confidence: {confidence:.2%})")
    logger.info(f"Estimated spread: {predicted_spread:+.1f} (home perspective)")
    
    # Get actual results (use the game_id from game_data which may be the matching one)
    actual_game_id = game_data.iloc[0]['game_id'] if len(game_data) > 0 else normalized_game_id
    actual_results = get_actual_results(actual_game_id)
    
    if actual_results and actual_results.get('home_score') is not None and actual_results.get('away_score') is not None:
        actual_home_score = int(actual_results['home_score'])
        actual_away_score = int(actual_results['away_score'])
        # Use actual home/away teams from results
        actual_home_team = actual_results.get('home_team', home_team)
        actual_away_team = actual_results.get('away_team', away_team)
        actual_winner = actual_home_team if actual_home_score > actual_away_score else actual_away_team
        actual_spread = calculate_spread(actual_home_score, actual_away_score)
        correct = (predicted_winner == actual_winner)
        
        logger.info(f"\nActual Results:")
        logger.info(f"  {actual_away_team} @ {actual_home_team}: {actual_away_score} - {actual_home_score}")
        logger.info(f"  Winner: {actual_winner}")
        logger.info(f"  Spread: {actual_spread:+.1f} (home perspective)")
        logger.info(f"\nPrediction {'✓ CORRECT' if correct else '✗ INCORRECT'}")
    else:
        actual_home_score = None
        actual_away_score = None
        actual_winner = None
        actual_spread = None
        correct = None
        logger.info("\nActual results not available (game may not have been played yet)")
    
    # Log result
    if log_path is None:
        log_path = Path(__file__).parent.parent / "logs" / "simulations" / "predictions_vs_actuals.log"
    
    log_prediction_result(
        log_path,
        normalized_game_id,
        predicted_winner,
        predicted_prob,
        predicted_spread,
        actual_winner,
        actual_home_score,
        actual_away_score,
        actual_spread,
        correct,
        home_team,
        away_team,
    )
    
    # Print match summary
    print("\n" + "=" * 80)
    print("PREDICTION SIMULATION SUMMARY")
    print("=" * 80)
    print(f"Game ID: {normalized_game_id}")
    print(f"Matchup: {away_team} @ {home_team}")
    print(f"Season: {season}, Week: {week}")
    print(f"\nPrediction:")
    print(f"  Winner: {predicted_winner}")
    print(f"  Confidence: {confidence:.2%}")
    print(f"  Probability (Home Win): {predicted_prob:.4f}")
    print(f"  Estimated Spread: {predicted_spread:+.1f}")
    
    if actual_results and actual_home_score is not None and actual_away_score is not None:
        actual_home_team = actual_results.get('home_team', home_team)
        actual_away_team = actual_results.get('away_team', away_team)
        print(f"\nActual Result:")
        print(f"  Score: {actual_away_team} {actual_away_score} - {actual_home_score} {actual_home_team}")
        print(f"  Winner: {actual_winner}")
        print(f"  Spread: {actual_spread:+.1f}")
        print(f"\nPrediction Accuracy: {'✓ CORRECT' if correct else '✗ INCORRECT'}")
        print(f"Spread Error: {abs(predicted_spread - actual_spread):.1f} points")
    else:
        print(f"\nActual Result: Not available (game may not have been played)")
    
    print(f"\nLogged to: {log_path}")
    print("=" * 80)
    
    return {
        'game_id': normalized_game_id,
        'home_team': home_team,
        'away_team': away_team,
        'predicted_winner': predicted_winner,
        'predicted_prob': predicted_prob,
        'predicted_spread': predicted_spread,
        'actual_winner': actual_winner,
        'actual_home_score': actual_home_score,
        'actual_away_score': actual_away_score,
        'actual_spread': actual_spread,
        'correct': correct,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Simulate real-world prediction for an NFL game",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict specific game
  python scripts/simulate_real_world_prediction.py --game-id 2025_WK14_DAL_DET
  
  # Use default game (Detroit @ Dallas, Week 14, 2025)
  python scripts/simulate_real_world_prediction.py
  
  # Use our format
  python scripts/simulate_real_world_prediction.py --game-id nfl_2025_14_DET_DAL
        """
    )
    parser.add_argument(
        '--game-id',
        type=str,
        default='2025_WK14_DAL_DET',
        help='Game ID in format 2025_WK14_DAL_DET or nfl_2025_14_DET_DAL (default: 2025_WK14_DAL_DET)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to ensemble model (default: uses v2 ensemble)'
    )
    parser.add_argument(
        '--feature-table',
        type=str,
        default='baseline',
        help='Feature table name (default: baseline)'
    )
    parser.add_argument(
        '--log-path',
        type=str,
        default=None,
        help='Path to log file (default: logs/simulations/predictions_vs_actuals.log)'
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path) if args.model_path else None
    log_path = Path(args.log_path) if args.log_path else None
    
    try:
        result = simulate_prediction(
            game_id=args.game_id,
            model_path=model_path,
            feature_table=args.feature_table,
            log_path=log_path,
        )
        return result
    except Exception as e:
        logger.error(f"Error simulating prediction: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

