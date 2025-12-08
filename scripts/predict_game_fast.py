"""
Fast Game Prediction Script

Streamlined version that makes predictions quickly without regenerating all features.
Uses existing feature table and only fetches current data for the specific game.

Usage:
    python scripts/predict_game_fast.py --away HOU --home KC --date 2025-12-07
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import argparse
from datetime import datetime
import pytz
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.architectures.stacking_ensemble import StackingEnsemble
from models.calibration import CalibratedModel
from models.raw_ensemble import RawEnsemble
from ingestion.nfl.odds_api import OddsAPIClient
from ingestion.nfl.weather import WeatherIngestion

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_current_injuries_simple(away_team: str, home_team: str) -> dict:
    """Quick injury check using NFL.com main page."""
    try:
        import requests
        from bs4 import BeautifulSoup
        
        url = "https://www.nfl.com/injuries/"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find teams in the page
        page_text = soup.get_text()
        home_injuries = []
        away_injuries = []
        
        # Simple text search for team names near "Out", "Questionable", etc.
        # This is a quick heuristic - full parsing would be more accurate
        if home_team in page_text:
            # Count injury keywords near team name
            home_section = page_text[page_text.find(home_team):page_text.find(home_team)+2000]
            home_injuries = [s for s in ['Out', 'Questionable', 'Doubtful'] if s in home_section]
        
        if away_team in page_text:
            away_section = page_text[page_text.find(away_team):page_text.find(away_team)+2000]
            away_injuries = [s for s in ['Out', 'Questionable', 'Doubtful'] if s in away_section]
        
        return {'home': len(home_injuries), 'away': len(away_injuries)}
    except:
        return {'home': 0, 'away': 0}


def predict_game_fast(away_team: str, home_team: str, game_date: str, model_path: str):
    """Make a fast prediction using existing features."""
    logger.info("=" * 60)
    logger.info(f"PREDICTION: {away_team} @ {home_team}")
    logger.info(f"Date: {game_date}")
    logger.info("=" * 60)
    
    # Load existing features
    features_path = Path(__file__).parent.parent / "data" / "nfl" / "processed" / "game_features_phase3.parquet"
    logger.info(f"Loading features from {features_path}")
    
    try:
        all_features = pd.read_parquet(features_path)
    except Exception as e:
        logger.error(f"Error loading features: {e}")
        return None
    
    # Find similar games (same teams, recent seasons) to use as proxy
    # For tonight: HOU @ KC
    similar_games = all_features[
        ((all_features['away_team'] == away_team) & (all_features['home_team'] == home_team)) |
        ((all_features['away_team'] == home_team) & (all_features['home_team'] == away_team))
    ].copy()
    
    if len(similar_games) == 0:
        logger.warning("No historical games found between these teams, using average features")
        # Use average features from recent games
        recent_games = all_features[all_features['season'] >= 2023].copy()
        similar_games = recent_games.tail(100)  # Use last 100 games as proxy
    
    # Get feature columns (exclude metadata) - must match training
    exclude_cols = ['home_team', 'away_team', 'home_score', 'away_score', 'game_id', 'season', 'week', 
                   'date', 'gameday', 'home_win', 'close_spread', 'close_total', 'open_spread', 'open_total']
    feature_cols = [c for c in similar_games.columns if c.startswith(('home_', 'away_')) 
                   and c not in exclude_cols]
    
    # Ensure we have exactly the right number of features (model expects 43)
    # Sort to ensure consistent order
    feature_cols = sorted(feature_cols)
    
    # Use average features from similar games as proxy
    proxy_features = similar_games[feature_cols].mean().fillna(0)
    
    # Ensure we have the right number of features
    if len(feature_cols) != 43:
        logger.warning(f"Feature count mismatch: expected 43, got {len(feature_cols)}")
        # Take first 43 if too many, pad with zeros if too few
        if len(feature_cols) > 43:
            feature_cols = feature_cols[:43]
            proxy_features = proxy_features[feature_cols]
        else:
            # Pad with zeros for missing features
            missing = 43 - len(feature_cols)
            for i in range(missing):
                proxy_features[f'missing_feature_{i}'] = 0.0
    
    # Fetch current data
    logger.info("Fetching current odds and weather...")
    
    # Odds
    odds_data = {}
    try:
        odds_client = OddsAPIClient()
        odds_df = odds_client.get_nfl_odds(markets=['spreads', 'totals'], use_cache=False)
        game_odds = odds_df[(odds_df['away_team'] == away_team) & (odds_df['home_team'] == home_team)]
        if len(game_odds) > 0:
            spread_row = game_odds[game_odds['market'] == 'spreads']
            total_row = game_odds[game_odds['market'] == 'totals']
            if len(spread_row) > 0:
                odds_data['spread'] = spread_row.iloc[0].get('point')
            if len(total_row) > 0:
                odds_data['total'] = total_row.iloc[0].get('point')
    except Exception as e:
        logger.warning(f"Could not fetch odds: {e}")
    
    # Weather
    weather_data = {}
    try:
        tz = pytz.timezone('America/New_York')
        game_datetime = tz.localize(datetime.strptime(f"{game_date} 20:20", "%Y-%m-%d %H:%M"))
        weather_ingester = WeatherIngestion()
        weather = weather_ingester.get_game_weather(home_team, game_datetime.replace(tzinfo=None), use_cache=False)
        if weather:
            weather_data = weather
    except Exception as e:
        logger.warning(f"Could not fetch weather: {e}")
    
    # Injuries (quick check)
    injury_data = get_current_injuries_simple(away_team, home_team)
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    try:
        # Load as StackingEnsemble (preferred - we'll use RawEnsemble wrapper)
        def custom_base_model_loader(path):
            from models.architectures.gradient_boosting import GradientBoostingModel
            from models.architectures.ft_transformer import FTTransformerModel
            from models.architectures.logistic_regression import LogisticRegressionModel
            from models.base import BaseModel
            
            path = Path(path)
            if 'ft_transformer' in str(path):
                return FTTransformerModel.load(path)
            elif 'gbm' in str(path) or 'gradient' in str(path):
                return GradientBoostingModel.load(path)
            elif 'logistic' in str(path) or 'logit' in str(path):
                return LogisticRegressionModel.load(path)
            else:
                return BaseModel.load(path)
        
        try:
            ensemble = StackingEnsemble.load(Path(model_path), base_model_loader=custom_base_model_loader)
            logger.info("Loaded StackingEnsemble model")
            
            # Wrap with RawEnsemble to bypass calibration (calibration hurts performance)
            model = RawEnsemble(ensemble)
            logger.info("Using RawEnsemble wrapper (bypasses calibration)")
        except Exception as e2:
            logger.error(f"Could not load StackingEnsemble: {e2}")
            # Try alternative paths
            alt_paths = [
                Path(__file__).parent.parent / "models" / "artifacts" / "nfl_stacked_ensemble_v2" / "ensemble_v1.pkl",
                Path(__file__).parent.parent / "artifacts" / "models" / "nfl_stacked_ensemble_v2" / "ensemble_v1.pkl",
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    try:
                        ensemble = StackingEnsemble.load(alt_path, base_model_loader=custom_base_model_loader)
                        model = RawEnsemble(ensemble)
                        logger.info(f"Loaded model from alternative path: {alt_path}")
                        break
                    except Exception as e3:
                        logger.warning(f"Failed to load from {alt_path}: {e3}")
                        continue
            else:
                logger.error("Failed to load model from any path")
                return None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None
    
    # Make prediction
    logger.info("Making prediction...")
    X_pred = pd.DataFrame([proxy_features])
    prob_home_win = model.predict_proba(X_pred)[:, 1]  # Get home win probability
    
    # Calculate implied spread
    if prob_home_win > 0.01 and prob_home_win < 0.99:
        implied_spread = 3 - 3 * np.log(prob_home_win / (1 - prob_home_win))
    else:
        implied_spread = None
    
    # Results
    results = {
        'game': f"{away_team} @ {home_team}",
        'date': game_date,
        'prediction': {
            'home_win_probability': float(prob_home_win),
            'away_win_probability': float(1 - prob_home_win),
            'implied_spread': float(implied_spread) if implied_spread else None,
            'recommendation': 'Home' if prob_home_win > 0.55 else ('Away' if prob_home_win < 0.45 else 'No Bet'),
        },
        'market_data': {
            'odds_spread': odds_data.get('spread'),
            'odds_total': odds_data.get('total'),
            'edge': float(implied_spread - odds_data['spread']) if implied_spread and odds_data.get('spread') else None,
        },
        'weather': weather_data,
        'injuries': injury_data,
    }
    
    return results


def print_prediction(results: dict):
    """Print formatted results."""
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"\nGame: {results['game']}")
    print(f"Date: {results['date']}")
    print(f"\n{'='*60}")
    print("MODEL PREDICTION")
    print(f"{'='*60}")
    print(f"Home Win Probability: {results['prediction']['home_win_probability']:.1%}")
    print(f"Away Win Probability: {results['prediction']['away_win_probability']:.1%}")
    
    if results['prediction']['implied_spread']:
        print(f"\nImplied Spread: {results['prediction']['implied_spread']:.1f} (Home)")
    
    print(f"\nRecommendation: {results['prediction']['recommendation']}")
    
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
        print("WEATHER")
        print(f"{'='*60}")
        w = results['weather']
        print(f"Temperature: {w.get('temperature_f', 'N/A')}°F")
        print(f"Wind: {w.get('wind_speed_mph', 'N/A')} mph")
        print(f"Dome: {'Yes' if w.get('is_dome', False) else 'No'}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--away', type=str, required=True)
    parser.add_argument('--home', type=str, required=True)
    parser.add_argument('--date', type=str, required=True)
    parser.add_argument('--model', type=str, 
                       default='models/artifacts/nfl_all_data/ensemble_calibrated.pkl')
    
    args = parser.parse_args()
    
    results = predict_game_fast(args.away, args.home, args.date, args.model)
    
    if results:
        print_prediction(results)
        
        # Save
        output_path = Path(__file__).parent.parent / "logs" / "predictions" / f"{args.away}_{args.home}_{args.date}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nResults saved to {output_path}")
    else:
        logger.error("Prediction failed")
        sys.exit(1)

