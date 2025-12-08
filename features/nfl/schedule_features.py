"""
NFL Schedule and Rest Features

Calculates rest-related and schedule context features for NFL games:
- Days of rest (since last game)
- Bye week indicators
- Short week indicators (Thursday games)
- Travel features (timezone differences)
- Schedule context (divisional games, primetime, playoff implications)

CRITICAL: All features use ONLY data available BEFORE the game (no leakage).
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


# NFL divisions (for divisional game detection)
NFL_DIVISIONS = {
    'AFC_EAST': ['BUF', 'MIA', 'NE', 'NYJ'],
    'AFC_NORTH': ['BAL', 'CIN', 'CLE', 'PIT'],
    'AFC_SOUTH': ['HOU', 'IND', 'JAX', 'TEN'],
    'AFC_WEST': ['DEN', 'KC', 'LV', 'LAC'],
    'NFC_EAST': ['DAL', 'NYG', 'PHI', 'WAS'],
    'NFC_NORTH': ['CHI', 'DET', 'GB', 'MIN'],
    'NFC_SOUTH': ['ATL', 'CAR', 'NO', 'TB'],
    'NFC_WEST': ['ARI', 'LAR', 'SF', 'SEA'],
}

# Team timezones (approximate, for travel calculations)
TEAM_TIMEZONES = {
    'BUF': 'America/New_York', 'MIA': 'America/New_York', 'NE': 'America/New_York', 'NYJ': 'America/New_York',
    'NYG': 'America/New_York', 'PHI': 'America/New_York', 'WAS': 'America/New_York',
    'BAL': 'America/New_York', 'CIN': 'America/New_York', 'CLE': 'America/New_York', 'PIT': 'America/New_York',
    'ATL': 'America/New_York', 'CAR': 'America/New_York', 'TB': 'America/New_York',
    'HOU': 'America/Chicago', 'IND': 'America/Indiana/Indianapolis', 'JAX': 'America/New_York', 'TEN': 'America/Chicago',
    'CHI': 'America/Chicago', 'DET': 'America/Detroit', 'GB': 'America/Chicago', 'MIN': 'America/Chicago',
    'DAL': 'America/Chicago', 'NO': 'America/Chicago',
    'DEN': 'America/Denver', 'KC': 'America/Chicago', 'LV': 'America/Los_Angeles', 'LAC': 'America/Los_Angeles',
    'ARI': 'America/Phoenix', 'LAR': 'America/Los_Angeles', 'SF': 'America/Los_Angeles', 'SEA': 'America/Los_Angeles',
}

# Timezone offsets (hours from UTC, approximate)
TIMEZONE_OFFSETS = {
    'America/New_York': -5,  # EST (adjust for DST in practice)
    'America/Chicago': -6,
    'America/Denver': -7,
    'America/Los_Angeles': -8,
    'America/Phoenix': -7,  # No DST
    'America/Detroit': -5,
    'America/Indiana/Indianapolis': -5,
}


def get_team_division(team: str) -> Optional[str]:
    """Get division for a team."""
    for division, teams in NFL_DIVISIONS.items():
        if team in teams:
            return division
    return None


def calculate_rest_features(
    games_df: pd.DataFrame,
    team: str,
    game_date: pd.Timestamp,
    season: int,
) -> Dict:
    """
    Calculate rest-related features for a team entering a game.
    
    Args:
        games_df: DataFrame with all games (must have columns: date, home_team, away_team, season)
        team: Team abbreviation
        game_date: Date of current game
        season: Season year
    
    Returns:
        Dictionary with rest features
    """
    # Filter to games for this team in this season (before current game)
    team_games = games_df[
        (games_df['season'] == season) &
        ((games_df['home_team'] == team) | (games_df['away_team'] == team)) &
        (games_df['date'] < game_date)
    ].sort_values('date')
    
    if len(team_games) == 0:
        # Season opener - no previous game
        return {
            'days_rest': np.nan,
            'is_short_week': 0,
            'is_bye_week_return': 0,
            'is_season_opener': 1,
        }
    
    # Get last game date
    last_game = team_games.iloc[-1]
    last_game_date = pd.to_datetime(last_game['date'])
    current_game_date = pd.to_datetime(game_date)
    
    # Calculate days of rest
    days_rest = (current_game_date - last_game_date).days
    
    # Short week: less than 6 days rest (Thursday games)
    is_short_week = 1 if days_rest < 6 else 0
    
    # Bye week return: 10+ days rest
    is_bye_week_return = 1 if days_rest >= 10 else 0
    
    return {
        'days_rest': days_rest,
        'is_short_week': is_short_week,
        'is_bye_week_return': is_bye_week_return,
        'is_season_opener': 0,
    }


def calculate_travel_features(
    home_team: str,
    away_team: str,
    game_date: pd.Timestamp,
) -> Dict:
    """
    Calculate travel-related features.
    
    Args:
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        game_date: Game date
    
    Returns:
        Dictionary with travel features
    """
    # Home team doesn't travel
    is_home = 1  # For home team
    is_away = 0  # For away team
    
    # Get timezones
    home_tz = TEAM_TIMEZONES.get(home_team, 'America/New_York')
    away_tz = TEAM_TIMEZONES.get(away_team, 'America/New_York')
    
    # Calculate timezone difference (simplified - doesn't account for DST)
    home_offset = TIMEZONE_OFFSETS.get(home_tz, -5)
    away_offset = TIMEZONE_OFFSETS.get(away_tz, -5)
    timezone_diff = abs(home_offset - away_offset)
    
    # Cross-country: 3+ timezone difference
    is_cross_country = 1 if timezone_diff >= 3 else 0
    
    return {
        'is_home': is_home,  # For home team
        'is_away': 0,  # For away team
        'travel_timezone_diff': timezone_diff,
        'is_cross_country': is_cross_country,
    }


def calculate_schedule_context_features(
    games_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    game_date: pd.Timestamp,
    season: int,
    week: int,
) -> Dict:
    """
    Calculate schedule context features.
    
    Args:
        games_df: DataFrame with all games
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        game_date: Game date
        season: Season year
        week: Week number
    
    Returns:
        Dictionary with schedule context features
    """
    # Divisional game
    home_division = get_team_division(home_team)
    away_division = get_team_division(away_team)
    is_divisional_game = 1 if home_division == away_division and home_division is not None else 0
    
    # Primetime games (simplified - check if game is on Thu/Sun/Mon)
    # In practice, you'd check actual game time, but this is a proxy
    day_of_week = game_date.dayofweek  # 0=Monday, 3=Thursday, 6=Sunday
    is_primetime = 1 if day_of_week in [0, 3, 6] else 0  # Mon, Thu, Sun
    
    # Week of season (early vs late)
    week_of_season = week
    
    # Playoff implications (late season, both teams in contention)
    # Simplified: weeks 14-18 are late season
    is_playoff_implication = 1 if week >= 14 else 0
    
    return {
        'is_divisional_game': is_divisional_game,
        'is_primetime': is_primetime,
        'week_of_season': week_of_season,
        'is_playoff_implication': is_playoff_implication,
    }


def calculate_consecutive_road_games(
    games_df: pd.DataFrame,
    team: str,
    game_date: pd.Timestamp,
    season: int,
) -> Dict:
    """
    Calculate consecutive road game features.
    
    Args:
        games_df: DataFrame with all games
        team: Team abbreviation
        game_date: Current game date
        season: Season year
    
    Returns:
        Dictionary with consecutive road game features
    """
    # Get team's games before current game
    team_games = games_df[
        (games_df['season'] == season) &
        ((games_df['home_team'] == team) | (games_df['away_team'] == team)) &
        (games_df['date'] < game_date)
    ].sort_values('date')
    
    if len(team_games) == 0:
        return {
            'consecutive_road_games': 0,
            'is_back_to_back_road': 0,
        }
    
    # Count consecutive road games (most recent first)
    consecutive_road = 0
    for _, game in team_games.iloc[::-1].iterrows():
        if game['away_team'] == team:
            consecutive_road += 1
        else:
            break
    
    is_back_to_back_road = 1 if consecutive_road >= 2 else 0
    
    return {
        'consecutive_road_games': consecutive_road,
        'is_back_to_back_road': is_back_to_back_road,
    }


def calculate_game_schedule_features(
    games_df: pd.DataFrame,
    game_id: str,
    home_team: str,
    away_team: str,
    game_date: pd.Timestamp,
    season: int,
    week: int,
) -> Dict:
    """
    Calculate all schedule features for a single game.
    
    This is the main function to call. It calculates features for both teams
    and returns a dictionary with all schedule-related features.
    
    CRITICAL: Only uses data from games BEFORE the current game (no leakage).
    
    Args:
        games_df: DataFrame with all games (must be sorted by date)
        game_id: Game ID
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        game_date: Game date
        season: Season year
        week: Week number
    
    Returns:
        Dictionary with all schedule features (prefixed with home_/away_)
    """
    features = {}
    
    # Home team rest features
    home_rest = calculate_rest_features(games_df, home_team, game_date, season)
    for key, value in home_rest.items():
        features[f'home_{key}'] = value
    
    # Away team rest features
    away_rest = calculate_rest_features(games_df, away_team, game_date, season)
    for key, value in away_rest.items():
        features[f'away_{key}'] = value
    
    # Rest advantage (home rest - away rest)
    if pd.notna(home_rest.get('days_rest')) and pd.notna(away_rest.get('days_rest')):
        features['rest_advantage'] = home_rest['days_rest'] - away_rest['days_rest']
    else:
        features['rest_advantage'] = np.nan
    
    # Travel features (for away team)
    travel_features = calculate_travel_features(home_team, away_team, game_date)
    features['away_travel_timezone_diff'] = travel_features['travel_timezone_diff']
    features['away_is_cross_country'] = travel_features['is_cross_country']
    
    # Consecutive road games (for away team)
    away_consecutive = calculate_consecutive_road_games(games_df, away_team, game_date, season)
    features['away_consecutive_road_games'] = away_consecutive['consecutive_road_games']
    features['away_is_back_to_back_road'] = away_consecutive['is_back_to_back_road']
    
    # Schedule context
    context_features = calculate_schedule_context_features(
        games_df, home_team, away_team, game_date, season, week
    )
    features.update(context_features)
    
    return features


def add_schedule_features_to_games(
    games_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add schedule features to a games DataFrame.
    
    This function processes all games and adds schedule features.
    It ensures no data leakage by only using games before each current game.
    
    Args:
        games_df: DataFrame with games (must have: game_id, date, home_team, away_team, season, week)
    
    Returns:
        DataFrame with schedule features added
    """
    logger.info("Calculating schedule features for all games...")
    
    # Ensure games are sorted by date
    games_df = games_df.sort_values(['season', 'week', 'date']).reset_index(drop=True)
    
    # Initialize feature columns
    feature_names = [
        'home_days_rest', 'home_is_short_week', 'home_is_bye_week_return', 'home_is_season_opener',
        'away_days_rest', 'away_is_short_week', 'away_is_bye_week_return', 'away_is_season_opener',
        'rest_advantage',
        'away_travel_timezone_diff', 'away_is_cross_country',
        'away_consecutive_road_games', 'away_is_back_to_back_road',
        'is_divisional_game', 'is_primetime', 'week_of_season', 'is_playoff_implication',
    ]
    
    for feature_name in feature_names:
        games_df[feature_name] = np.nan
    
    # Calculate features for each game
    for idx, row in games_df.iterrows():
        try:
            features = calculate_game_schedule_features(
                games_df,
                row['game_id'],
                row['home_team'],
                row['away_team'],
                row['date'],
                row['season'],
                row['week'],
            )
            
            # Add features to DataFrame
            for feature_name, value in features.items():
                games_df.at[idx, feature_name] = value
            
        except Exception as e:
            logger.warning(f"Error calculating features for game {row['game_id']}: {e}")
            continue
    
    logger.info(f"Added schedule features to {len(games_df)} games")
    
    return games_df

