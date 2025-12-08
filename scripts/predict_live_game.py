"""
Real-World Game Prediction Script

Makes predictions for live/upcoming games using the trained ensemble model.
Fetches current odds, weather, and injury data for accurate predictions.

Usage:
    python scripts/predict_live_game.py --away HOU --home KC --date 2025-12-07 --time "20:20"
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import argparse
from datetime import datetime
from typing import Optional
import pytz

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.nfl.generate_all_features import generate_all_features
from models.architectures.stacking_ensemble import StackingEnsemble
from models.calibration import CalibratedModel
from ingestion.nfl.odds_api import OddsAPIClient
from ingestion.nfl.weather import WeatherIngestion
from ingestion.nfl.injuries_phase2 import InjuryIngestion
from features.nfl.schedule_features import add_schedule_features_to_games
from features.nfl.injury_features import add_injury_features_to_games
from features.nfl.weather_features import add_weather_features_to_games

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_game_row(away_team: str, home_team: str, game_date: datetime, season: int, week: int) -> pd.DataFrame:
    """Create a single-row DataFrame for the game."""
    game_id = f"nfl_{season}_{week:02d}_{away_team}_{home_team}"
    
    game_row = pd.DataFrame([{
        'game_id': game_id,
        'season': season,
        'week': week,
        'gameday': game_date.date(),
        'away_team': away_team,
        'home_team': home_team,
        'home_score': None,  # Game hasn't been played yet
        'away_score': None,
        'game_type': 'REG',  # Regular season
    }])
    
    return game_row


def fetch_live_data(away_team: str, home_team: str, game_datetime: datetime) -> dict:
    """Fetch current odds, weather, and injury data for the game."""
    live_data = {
        'odds': None,
        'weather': None,
        'injuries': None,
    }
    
    # Fetch odds
    try:
        logger.info("Fetching current odds...")
        odds_client = OddsAPIClient()
        odds_df = odds_client.get_nfl_odds(markets=['spreads', 'totals'], use_cache=False)
        
        # Filter to our game
        game_odds = odds_df[
            (odds_df['away_team'] == away_team) & 
            (odds_df['home_team'] == home_team)
        ]
        
        if len(game_odds) > 0:
            # Get spread and total
            spread_row = game_odds[game_odds['market'] == 'spreads']
            total_row = game_odds[game_odds['market'] == 'totals']
            
            odds_dict = {}
            if len(spread_row) > 0:
                odds_dict['spread'] = spread_row.iloc[0].get('point')
            if len(total_row) > 0:
                odds_dict['total'] = total_row.iloc[0].get('point')
            
            if odds_dict:
                live_data['odds'] = odds_dict
                logger.info(f"  Found odds: {odds_dict}")
            else:
                logger.warning("  No odds found")
        else:
            logger.warning("  No odds found for this game")
    except Exception as e:
        logger.warning(f"  Error fetching odds: {e}")
    
    # Fetch weather
    try:
        logger.info("Fetching weather data...")
        weather_ingester = WeatherIngestion()
        # Convert to naive datetime for weather API
        game_datetime_naive = game_datetime.astimezone(pytz.UTC).replace(tzinfo=None)
        weather = weather_ingester.get_game_weather(home_team, game_datetime_naive, use_cache=False)
        if weather:
            live_data['weather'] = weather
            logger.info(f"  Weather: Temp={weather.get('temperature_f')}°F, Wind={weather.get('wind_speed_mph')}mph")
        else:
            logger.warning("  No weather data found")
    except Exception as e:
        logger.warning(f"  Error fetching weather: {e}")
    
    # Fetch injuries (try current injuries API first, not historical)
    try:
        logger.info("Fetching current injury data...")
        injury_ingester = InjuryIngestion(source='auto')
        
        # Try to fetch current injuries (for live game)
        try:
            current_injuries = injury_ingester.fetch_current_injuries()
            if len(current_injuries) > 0:
                # Filter to our teams
                injuries_home = current_injuries[current_injuries['team'] == home_team]
                injuries_away = current_injuries[current_injuries['team'] == away_team]
                
                live_data['injuries'] = {
                    'home': injuries_home.to_dict('records') if len(injuries_home) > 0 else [],
                    'away': injuries_away.to_dict('records') if len(injuries_away) > 0 else [],
                }
                logger.info(f"  Current injuries: Home={len(injuries_home)}, Away={len(injuries_away)}")
            else:
                logger.warning("  No current injury data available")
        except Exception as e:
            logger.warning(f"  Could not fetch current injuries: {e}")
            logger.info("  Will try historical injuries for this week...")
            # Fallback: try historical injuries for this week
            injuries_home = injury_ingester.get_team_injuries(home_team, week, season)
            injuries_away = injury_ingester.get_team_injuries(away_team, week, season)
            
            if len(injuries_home) > 0 or len(injuries_away) > 0:
                live_data['injuries'] = {
                    'home': injuries_home.to_dict('records') if len(injuries_home) > 0 else [],
                    'away': injuries_away.to_dict('records') if len(injuries_away) > 0 else [],
                }
                logger.info(f"  Historical injuries: Home={len(injuries_home)}, Away={len(injuries_away)}")
    except Exception as e:
        logger.warning(f"  Error fetching injuries: {e}")
    
    return live_data


def predict_game(
    away_team: str,
    home_team: str,
    game_date: str,
    game_time: str = "20:20",
    timezone: str = "America/New_York",
    model_path: Optional[str] = None,
) -> dict:
    """
    Make a prediction for a single game.
    
    Args:
        away_team: Away team abbreviation (e.g., 'HOU')
        home_team: Home team abbreviation (e.g., 'KC')
        game_date: Game date in YYYY-MM-DD format
        game_time: Game time in HH:MM format (24-hour)
        timezone: Timezone for game time (default: America/New_York)
        model_path: Path to trained model (default: uses latest)
    
    Returns:
        Dictionary with prediction results
    """
    logger.info("=" * 60)
    logger.info("REAL-WORLD GAME PREDICTION")
    logger.info("=" * 60)
    logger.info(f"Game: {away_team} @ {home_team}")
    logger.info(f"Date: {game_date} {game_time} {timezone}")
    
    # Parse game datetime
    tz = pytz.timezone(timezone)
    game_datetime_str = f"{game_date} {game_time}"
    game_datetime = tz.localize(datetime.strptime(game_datetime_str, "%Y-%m-%d %H:%M"))
    
    # Determine season and week
    season = int(game_date[:4])
    # For now, assume week 14 (December 7, 2025 would be around week 14)
    # In production, this should be calculated from the date
    week = 14  # TODO: Calculate from date
    
    # Create game row
    game_row = create_game_row(away_team, home_team, game_datetime, season, week)
    
    # Load all historical games for feature generation
    logger.info("Loading historical game data...")
    games_path = Path(__file__).parent.parent / "data" / "nfl" / "staged" / "games.parquet"
    all_games = pd.read_parquet(games_path)
    
    # Filter to completed games only (for training context)
    completed_games = all_games[all_games['home_score'].notna() & all_games['away_score'].notna()].copy()
    
    # Combine with our game (for feature generation context)
    games_with_prediction = pd.concat([completed_games, game_row], ignore_index=True)
    
    # Fetch live data
    logger.info("Fetching live data (odds, weather, injuries)...")
    live_data = fetch_live_data(away_team, home_team, game_datetime)
    
    # Generate features for the prediction game
    logger.info("Generating features...")
    try:
        # Add our game to the games dataframe temporarily
        # Save original games, add our game, generate features, then restore
        games_path = Path(__file__).parent.parent / "data" / "nfl" / "staged" / "games.parquet"
        games_df_orig = pd.read_parquet(games_path)
        
        # Add our game temporarily
        games_with_prediction = pd.concat([games_df_orig, game_row], ignore_index=True)
        
        # Save temporarily
        temp_games_path = games_path.parent / "games_temp.parquet"
        games_with_prediction.to_parquet(temp_games_path, index=False)
        
        # Generate features (this will load from games.parquet)
        # We need to temporarily replace the games file
        import shutil
        backup_path = games_path.parent / "games_backup.parquet"
        shutil.copy(games_path, backup_path)
        shutil.copy(temp_games_path, games_path)
        
        try:
            # Generate features
            # For live prediction, try to use real data (not mock)
            # Note: If real injury data isn't available, we'll skip injury features
            # rather than using mock data for a live game prediction
            logger.info("Generating features with real data only (no mock data)...")
            features_df = generate_all_features(
                use_mock_injuries=False,  # Don't use mock injury data for live game
                use_weather_cache=False,  # Use fresh weather for live game
            )
            
            # Filter to just our prediction game
            prediction_row = features_df[features_df['game_id'] == game_row.iloc[0]['game_id']].copy()
            
            if len(prediction_row) == 0:
                raise ValueError("Prediction game not found in features")
            
            logger.info(f"Generated {len([c for c in prediction_row.columns if c.startswith(('home_', 'away_'))])} features")
        finally:
            # Restore original games file
            shutil.copy(backup_path, games_path)
            backup_path.unlink()
            temp_games_path.unlink()
        
    except Exception as e:
        logger.error(f"Error generating features: {e}")
        raise
    
    # Load trained model
    if model_path is None:
        model_path = Path(__file__).parent.parent / "models" / "artifacts" / "nfl_phase3" / "ensemble_calibrated.pkl"
    
    logger.info(f"Loading model from {model_path}")
    from models.raw_ensemble import RawEnsemble
    
    # Load ensemble and wrap with RawEnsemble to bypass calibration
    ensemble = StackingEnsemble.load(model_path)
    ensemble = RawEnsemble(ensemble)  # Use raw predictions (calibration hurts performance)
    
    # Get feature columns (same as training)
    feature_cols = [c for c in prediction_row.columns if c.startswith(('home_', 'away_')) and c not in ['home_team', 'away_team', 'home_score', 'away_score']]
    
    # Make prediction
    logger.info("Making prediction...")
    X_pred = prediction_row[feature_cols].fillna(0)
    prob_home_win = ensemble.predict_proba(X_pred)[:, 1]  # Get home win probability
    
    # Calculate implied spread from probability
    # Using logistic relationship: prob = 1 / (1 + exp(-(spread - 3) / 3))
    # Solving for spread: spread = 3 - 3 * log(prob / (1 - prob))
    if prob_home_win > 0.01 and prob_home_win < 0.99:
        implied_spread = 3 - 3 * np.log(prob_home_win / (1 - prob_home_win))
    else:
        implied_spread = np.nan
    
    # Prepare results
    results = {
        'game': f"{away_team} @ {home_team}",
        'date': game_date,
        'time': game_time,
        'prediction': {
            'home_win_probability': float(prob_home_win),
            'away_win_probability': float(1 - prob_home_win),
            'implied_spread': float(implied_spread) if not np.isnan(implied_spread) else None,
            'recommended_bet': 'Home' if prob_home_win > 0.55 else ('Away' if prob_home_win < 0.45 else 'No Bet'),
        },
        'market_data': {
            'odds_spread': live_data['odds'].get('spread') if live_data['odds'] else None,
            'odds_total': live_data['odds'].get('total') if live_data['odds'] else None,
            'edge': float(implied_spread - live_data['odds']['spread']) if live_data['odds'] and live_data['odds'].get('spread') and not np.isnan(implied_spread) else None,
        },
        'weather': live_data['weather'],
        'injuries': live_data['injuries'],
    }
    
    return results


def print_prediction(results: dict):
    """Print formatted prediction results."""
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"\nGame: {results['game']}")
    print(f"Date: {results['date']} {results['time']}")
    print(f"\n{'='*60}")
    print("MODEL PREDICTION")
    print(f"{'='*60}")
    print(f"Home Win Probability: {results['prediction']['home_win_probability']:.1%}")
    print(f"Away Win Probability: {results['prediction']['away_win_probability']:.1%}")
    
    if results['prediction']['implied_spread']:
        print(f"\nImplied Spread: {results['prediction']['implied_spread']:.1f} (Home)")
    
    print(f"\nRecommendation: {results['prediction']['recommended_bet']}")
    
    if results['market_data']['odds_spread']:
        print(f"\n{'='*60}")
        print("MARKET COMPARISON")
        print(f"{'='*60}")
        print(f"Market Spread: {results['market_data']['odds_spread']:.1f} (Home)")
        if results['prediction']['implied_spread']:
            print(f"Model Spread: {results['prediction']['implied_spread']:.1f} (Home)")
            if results['market_data']['edge']:
                edge = results['market_data']['edge']
                print(f"Edge: {edge:+.1f} points")
                if abs(edge) > 1.0:
                    print(f"  → {'VALUE BET' if edge > 0 else 'FADE BET'}")
    
    if results['weather']:
        print(f"\n{'='*60}")
        print("WEATHER CONDITIONS")
        print(f"{'='*60}")
        weather = results['weather']
        print(f"Temperature: {weather.get('temperature_f', 'N/A')}°F")
        print(f"Wind Speed: {weather.get('wind_speed_mph', 'N/A')} mph")
        print(f"Precipitation: {weather.get('precipitation_inches', 0):.2f}\"")
        print(f"Dome: {'Yes' if weather.get('is_dome', False) else 'No'}")
    
    if results['injuries']:
        print(f"\n{'='*60}")
        print("INJURY STATUS")
        print(f"{'='*60}")
        injuries = results['injuries']
        print(f"Home Team Injuries: {len(injuries.get('home', []))}")
        print(f"Away Team Injuries: {len(injuries.get('away', []))}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Predict NFL game outcome")
    parser.add_argument('--away', type=str, required=True, help='Away team abbreviation (e.g., HOU)')
    parser.add_argument('--home', type=str, required=True, help='Home team abbreviation (e.g., KC)')
    parser.add_argument('--date', type=str, required=True, help='Game date (YYYY-MM-DD)')
    parser.add_argument('--time', type=str, default='20:20', help='Game time (HH:MM, 24-hour)')
    parser.add_argument('--timezone', type=str, default='America/New_York', help='Timezone')
    parser.add_argument('--model', type=str, help='Path to model file')
    
    args = parser.parse_args()
    
    try:
        results = predict_game(
            away_team=args.away,
            home_team=args.home,
            game_date=args.date,
            game_time=args.time,
            timezone=args.timezone,
            model_path=args.model,
        )
        
        print_prediction(results)
        
        # Save results
        output_path = Path(__file__).parent.parent / "logs" / "predictions" / f"{args.away}_{args.home}_{args.date}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

