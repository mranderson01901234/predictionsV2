"""
NFL Weather Features

Calculates weather-related features for NFL game predictions.
Weather primarily affects:
1. Passing game (wind, precipitation)
2. Kicking game (wind)
3. Player performance (extreme temps)

Only applies to outdoor games (~50% of games).
Dome/indoor stadiums have neutral weather features.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_weather_features(weather_data: Dict) -> Dict:
    """
    Calculate weather-related features for a game.
    
    Args:
        weather_data: Dictionary from WeatherIngestion.get_game_weather()
    
    Returns:
        {
            # Binary indicators
            'is_dome': 0/1,
            'is_cold_game': 0/1,        # < 32°F
            'is_freezing_game': 0/1,    # < 20°F
            'is_hot_game': 0/1,         # > 85°F
            'is_windy': 0/1,            # > 15 mph
            'is_very_windy': 0/1,       # > 25 mph
            'is_precipitation': 0/1,    # Rain/snow likely
            
            # Continuous features
            'temperature_f': float,
            'wind_speed_mph': float,
            'precipitation_prob': float,
            
            # Composite scores
            'passing_conditions_score': float,  # 0-100, higher = better for passing
            'kicking_conditions_score': float,  # 0-100, higher = better for kicking
        }
    """
    if weather_data.get('is_dome'):
        return _get_neutral_weather_features()
    
    temp = weather_data.get('temperature_f', 60)
    wind = weather_data.get('wind_speed_mph', 0)
    precip_prob = weather_data.get('precipitation_prob', 0)
    precip_inches = weather_data.get('precipitation_inches', 0)
    
    # Calculate passing conditions (wind and precipitation hurt passing)
    passing_score = 100.0
    passing_score -= min(wind * 2, 40)  # Wind penalty (max -40)
    passing_score -= min(precip_prob * 0.3, 30)  # Precipitation probability penalty (max -30)
    if precip_inches > 0:
        passing_score -= min(precip_inches * 10, 20)  # Actual precipitation penalty (max -20)
    passing_score = max(passing_score, 0)
    
    # Calculate kicking conditions (wind is primary factor)
    kicking_score = 100.0
    kicking_score -= min(wind * 3, 60)  # Wind penalty (max -60)
    kicking_score -= min(precip_prob * 0.2, 20)  # Precipitation penalty (max -20)
    if precip_inches > 0:
        kicking_score -= min(precip_inches * 15, 30)  # Actual precipitation penalty (max -30)
    kicking_score = max(kicking_score, 0)
    
    return {
        'is_dome': 0,
        'is_cold_game': 1 if temp < 32 else 0,
        'is_freezing_game': 1 if temp < 20 else 0,
        'is_hot_game': 1 if temp > 85 else 0,
        'is_windy': 1 if wind > 15 else 0,
        'is_very_windy': 1 if wind > 25 else 0,
        'is_precipitation': 1 if precip_prob > 50 or precip_inches > 0.01 else 0,
        'temperature_f': float(temp),
        'wind_speed_mph': float(wind),
        'precipitation_prob': float(precip_prob),
        'passing_conditions_score': float(passing_score),
        'kicking_conditions_score': float(kicking_score),
    }


def calculate_weather_matchup_features(
    weather: Dict,
    home_team: str,
    away_team: str,
    home_team_stats: Optional[Dict] = None,
    away_team_stats: Optional[Dict] = None,
) -> Dict:
    """
    Calculate team-specific weather advantages.
    
    Some teams are built for bad weather (run-heavy, strong defense).
    Some teams rely on passing (hurt by wind/precipitation).
    
    Args:
        weather: Weather features dict
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        home_team_stats: Optional team stats (for dynamic calculation)
        away_team_stats: Optional team stats (for dynamic calculation)
    
    Returns:
        {
            'home_weather_advantage': float,  # Home team's weather acclimation
            'pass_heavy_team_disadvantage': float,  # For pass-first teams in bad weather
        }
    """
    # Teams that play in cold/bad weather regularly (acclimated)
    COLD_WEATHER_TEAMS = ['BUF', 'GB', 'CHI', 'NE', 'NYG', 'NYJ', 'CLE', 'PIT', 'DEN', 'KC', 'MIN']
    
    # Teams with historically pass-heavy offenses (hurt by wind/precipitation)
    # This would ideally be calculated from team stats, not hardcoded
    PASS_HEAVY_TEAMS = ['KC', 'LAC', 'BUF', 'CIN', 'MIA', 'DET', 'SF', 'DAL']
    
    home_weather_advantage = 0.0
    
    # Home team acclimation advantage
    if home_team in COLD_WEATHER_TEAMS:
        if weather.get('is_cold_game', 0) or weather.get('is_freezing_game', 0):
            home_weather_advantage += 0.05  # Small advantage in cold weather
    
    # Away team disadvantage in extreme weather
    if away_team not in COLD_WEATHER_TEAMS:
        if weather.get('is_cold_game', 0) or weather.get('is_freezing_game', 0):
            home_weather_advantage += 0.03  # Away team less acclimated
    
    # Pass-heavy team disadvantage in bad weather
    pass_heavy_disadvantage = 0.0
    if weather.get('is_windy', 0) or weather.get('is_precipitation', 0):
        if home_team in PASS_HEAVY_TEAMS:
            pass_heavy_disadvantage -= 0.03
        if away_team in PASS_HEAVY_TEAMS:
            pass_heavy_disadvantage += 0.03  # Advantage for home team if away team is pass-heavy
    
    return {
        'home_weather_advantage': float(home_weather_advantage),
        'pass_heavy_team_disadvantage': float(pass_heavy_disadvantage),
    }


def _get_neutral_weather_features() -> Dict:
    """Return neutral weather features for dome/indoor games."""
    return {
        'is_dome': 1,
        'is_cold_game': 0,
        'is_freezing_game': 0,
        'is_hot_game': 0,
        'is_windy': 0,
        'is_very_windy': 0,
        'is_precipitation': 0,
        'temperature_f': 72.0,
        'wind_speed_mph': 0.0,
        'precipitation_prob': 0.0,
        'passing_conditions_score': 100.0,
        'kicking_conditions_score': 100.0,
    }


def add_weather_features_to_games(
    games_df: pd.DataFrame,
    weather_df: Optional[pd.DataFrame] = None,
    weather_ingester: Optional[object] = None,
) -> pd.DataFrame:
    """
    Add weather features to a games DataFrame.
    
    Args:
        games_df: DataFrame with games (must have: game_id, home_team, gameday/date)
        weather_df: Optional DataFrame with pre-fetched weather data
        weather_ingester: Optional WeatherIngestion instance for fetching weather
    
    Returns:
        DataFrame with weather features added
    """
    logger.info("Adding weather features to games...")
    
    # Initialize weather feature columns
    weather_feature_names = [
        'is_dome', 'is_cold_game', 'is_freezing_game', 'is_hot_game',
        'is_windy', 'is_very_windy', 'is_precipitation',
        'temperature_f', 'wind_speed_mph', 'precipitation_prob',
        'passing_conditions_score', 'kicking_conditions_score',
        'home_weather_advantage', 'pass_heavy_team_disadvantage',
    ]
    
    for feature_name in weather_feature_names:
        games_df[feature_name] = np.nan
    
    # If weather_df provided, merge it
    if weather_df is not None and len(weather_df) > 0:
        # Merge on game_id or home_team + date
        merge_cols = ['game_id'] if 'game_id' in weather_df.columns else ['home_team', 'season', 'week']
        
        for col in weather_feature_names:
            if col in weather_df.columns:
                games_df = games_df.merge(
                    weather_df[merge_cols + [col]],
                    on=merge_cols,
                    how='left',
                    suffixes=('', '_weather')
                )
                # Use weather value if available
                if f'{col}_weather' in games_df.columns:
                    games_df[col] = games_df[col].fillna(games_df[f'{col}_weather'])
                    games_df = games_df.drop(columns=[f'{col}_weather'])
    
    # Calculate features for each game
    for idx, row in games_df.iterrows():
        try:
            # Check if already has weather data
            if pd.notna(row.get('temperature_f')):
                # Weather data already present, calculate features
                weather_data = {
                    'is_dome': row.get('is_dome', 0),
                    'temperature_f': row.get('temperature_f', 60),
                    'wind_speed_mph': row.get('wind_speed_mph', 0),
                    'precipitation_prob': row.get('precipitation_prob', 0),
                    'precipitation_inches': row.get('precipitation_inches', 0),
                }
            else:
                # Need to fetch weather
                if weather_ingester is None:
                    from ingestion.nfl.weather import WeatherIngestion
                    weather_ingester = WeatherIngestion()
                
                home_team = row['home_team']
                game_date = row.get('gameday') or row.get('date')
                
                if pd.isna(game_date):
                    # Use default weather
                    weather_data = weather_ingester._get_default_weather()
                else:
                    if isinstance(game_date, str):
                        game_datetime = pd.to_datetime(game_date)
                    else:
                        game_datetime = game_date
                    
                    weather_data = weather_ingester.get_game_weather(home_team, game_datetime)
            
            # Calculate features
            features = calculate_weather_features(weather_data)
            
            # Calculate matchup features
            matchup_features = calculate_weather_matchup_features(
                features,
                row['home_team'],
                row['away_team'],
            )
            
            # Add features to DataFrame
            for feature_name, value in features.items():
                games_df.at[idx, feature_name] = value
            
            for feature_name, value in matchup_features.items():
                games_df.at[idx, feature_name] = value
            
        except Exception as e:
            logger.warning(f"Error calculating weather features for game {row.get('game_id', 'unknown')}: {e}")
            # Fill with neutral values
            neutral = _get_neutral_weather_features()
            for feature_name in weather_feature_names:
                if games_df.at[idx, feature_name] is np.nan:
                    games_df.at[idx, feature_name] = neutral.get(feature_name, 0)
    
    logger.info(f"Added weather features to {len(games_df)} games")
    return games_df

