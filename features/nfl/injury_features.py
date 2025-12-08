"""
NFL Injury Features

Calculates injury-related features for NFL game predictions:
- Position-weighted injury impact scores
- QB injury status (most critical)
- Offensive line health
- Skill position injuries
- Secondary injuries
- Injury advantage/disadvantage

CRITICAL: Uses injury status AS OF game day (not after).
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


# Position Impact Weights
POSITION_WEIGHTS = {
    # Offense
    'QB': 10.0,    # Most important position by far
    'LT': 3.5,     # Protects QB blind side
    'RT': 2.5,
    'LG': 2.0,
    'RG': 2.0,
    'C': 2.5,
    'WR': 2.0,     # Depends on depth
    'RB': 1.5,
    'TE': 1.5,
    'FB': 0.5,
    
    # Defense
    'EDGE': 2.5,   # Pass rushers
    'DT': 2.0,
    'DE': 2.0,
    'LB': 1.5,
    'CB': 2.5,     # Cover receivers
    'S': 1.5,
    
    # Special Teams
    'K': 1.0,
    'P': 0.5,
    'LS': 0.3,
}

# O-line positions
OLINE_POSITIONS = ['LT', 'RT', 'LG', 'RG', 'C', 'OL', 'OT', 'OG']

# Skill positions
SKILL_POSITIONS = ['WR', 'RB', 'TE', 'FB']

# Secondary positions
SECONDARY_POSITIONS = ['CB', 'S', 'DB', 'FS', 'SS']


def get_position_weight(position: str) -> float:
    """
    Get impact weight for a position.
    
    Args:
        position: Player position abbreviation
    
    Returns:
        Weight value (default 1.0 if position not found)
    """
    position_upper = position.upper().strip()
    
    # Direct match
    if position_upper in POSITION_WEIGHTS:
        return POSITION_WEIGHTS[position_upper]
    
    # Handle variations
    if position_upper in ['OL', 'OT', 'OG']:
        # Generic O-line - use average O-line weight
        return 2.5
    
    if position_upper in ['DB', 'FS', 'SS']:
        # Generic secondary - use average secondary weight
        return 2.0
    
    if position_upper in ['DL', 'NT']:
        # Generic D-line
        return 2.0
    
    # Default weight for unknown positions
    return 1.0


def calculate_qb_injury_status(injuries_df: pd.DataFrame, team: str) -> int:
    """
    Determine starting QB status.
    
    Returns:
        - 0: Starting QB healthy/full practice
        - 1: Starting QB questionable/limited
        - 2: Starting QB out (backup starting)
    
    This is the single most important injury feature.
    """
    if injuries_df is None or len(injuries_df) == 0:
        return 0  # No injury data = assume healthy
    
    # Filter to QB injuries for this team
    team_qb_injuries = injuries_df[
        (injuries_df['team'] == team) &
        (injuries_df['position'].str.upper() == 'QB')
    ].copy()
    
    if len(team_qb_injuries) == 0:
        return 0  # No QB injuries = healthy
    
    # Check game status (most important)
    game_statuses = team_qb_injuries['game_status'].str.upper().values
    
    if 'OUT' in game_statuses:
        return 2  # QB out
    
    if any(s in ['DOUBTFUL', 'QUESTIONABLE'] for s in game_statuses):
        return 1  # QB questionable
    
    # Check practice status
    practice_statuses = team_qb_injuries['practice_status'].str.upper().values
    
    if 'DNP' in practice_statuses or 'OUT' in practice_statuses:
        return 2  # Didn't practice = likely out
    
    if 'LIMITED' in practice_statuses:
        return 1  # Limited practice = questionable
    
    return 0  # Full practice = healthy


def calculate_oline_health(injuries_df: pd.DataFrame, team: str) -> Dict:
    """
    Calculate offensive line health score.
    
    O-line injuries compound — multiple injuries worse than sum of parts.
    
    Returns:
        {
            'oline_injuries_count': int,
            'oline_health_score': float,  # 0-100 scale (100 = fully healthy)
            'oline_is_compromised': int,  # 1 if 2+ starters out
        }
    """
    if injuries_df is None or len(injuries_df) == 0:
        return {
            'oline_injuries_count': 0,
            'oline_health_score': 100.0,
            'oline_is_compromised': 0,
        }
    
    # Filter to O-line injuries for this team
    team_oline = injuries_df[
        (injuries_df['team'] == team) &
        (injuries_df['position'].str.upper().isin(OLINE_POSITIONS))
    ].copy()
    
    if len(team_oline) == 0:
        return {
            'oline_injuries_count': 0,
            'oline_health_score': 100.0,
            'oline_is_compromised': 0,
        }
    
    # Count injuries by severity
    out_count = (team_oline['game_status'].str.upper() == 'OUT').sum()
    doubtful_count = (team_oline['game_status'].str.upper() == 'DOUBTFUL').sum()
    questionable_count = (team_oline['game_status'].str.upper() == 'QUESTIONABLE').sum()
    
    total_injuries = len(team_oline)
    
    # Calculate health score (0-100)
    # Each out = -20 points, doubtful = -10, questionable = -5
    health_score = 100.0
    health_score -= out_count * 20
    health_score -= doubtful_count * 10
    health_score -= questionable_count * 5
    
    # Compound penalty: multiple injuries worse than sum
    if total_injuries >= 2:
        health_score -= (total_injuries - 1) * 5  # Additional penalty
    
    health_score = max(0.0, min(100.0, health_score))
    
    # Compromised if 2+ starters out
    is_compromised = 1 if out_count >= 2 else 0
    
    return {
        'oline_injuries_count': total_injuries,
        'oline_health_score': health_score,
        'oline_is_compromised': is_compromised,
    }


def calculate_injury_features(
    injuries_df: pd.DataFrame,
    team: str,
    opponent: str,
    week: int,
    season: int,
) -> Dict:
    """
    Calculate injury-related features for a matchup.
    
    Args:
        injuries_df: DataFrame with injury data (must have columns: team, position, game_status, practice_status)
        team: Team abbreviation
        opponent: Opponent team abbreviation
        week: Week number
        season: Season year
    
    Returns:
        Dictionary with injury features:
        - team_players_out: Count of players with 'Out' status
        - team_players_questionable: Count of 'Questionable' players
        - team_weighted_injury_impact: Position-weighted injury score
        - team_qb_status: 0=healthy, 1=questionable, 2=out
        - team_oline_injuries: Count of O-line injuries
        - team_skill_position_injuries: WR/RB/TE injuries
        - team_secondary_injuries: CB/S injuries
        - opponent_* : Same features for opponent
        - injury_advantage: team_impact - opponent_impact (negative = disadvantage)
    """
    # Filter injuries to this week/season
    if injuries_df is not None and len(injuries_df) > 0:
        week_injuries = injuries_df[
            (injuries_df['week'] == week) &
            (injuries_df['season'] == season)
        ].copy()
    else:
        week_injuries = pd.DataFrame()
    
    # Team features
    team_features = _calculate_team_injury_features(week_injuries, team)
    
    # Opponent features
    opponent_features = _calculate_team_injury_features(week_injuries, opponent)
    
    # Combine features
    features = {}
    
    # Team features (prefix with team_)
    for key, value in team_features.items():
        features[f'team_{key}'] = value
    
    # Opponent features (prefix with opponent_)
    for key, value in opponent_features.items():
        features[f'opponent_{key}'] = value
    
    # Injury advantage (negative = disadvantage)
    team_impact = team_features.get('weighted_injury_impact', 0)
    opponent_impact = opponent_features.get('weighted_injury_impact', 0)
    features['injury_advantage'] = team_impact - opponent_impact
    
    return features


def _calculate_team_injury_features(injuries_df: pd.DataFrame, team: str) -> Dict:
    """
    Calculate injury features for a single team.
    
    Returns:
        Dictionary with team injury features
    """
    if injuries_df is None or len(injuries_df) == 0:
        return _get_default_injury_features()
    
    # Filter to team injuries
    team_injuries = injuries_df[injuries_df['team'] == team].copy()
    
    if len(team_injuries) == 0:
        return _get_default_injury_features()
    
    # Count by status
    players_out = (team_injuries['game_status'].str.upper() == 'OUT').sum()
    players_questionable = (team_injuries['game_status'].str.upper() == 'QUESTIONABLE').sum()
    players_doubtful = (team_injuries['game_status'].str.upper() == 'DOUBTFUL').sum()
    
    # Calculate weighted injury impact
    weighted_impact = 0.0
    for _, injury in team_injuries.iterrows():
        position = injury.get('position', '')
        game_status = injury.get('game_status', '').upper()
        
        weight = get_position_weight(position)
        
        # Status multipliers
        if game_status == 'OUT':
            weighted_impact += weight * 1.0
        elif game_status == 'DOUBTFUL':
            weighted_impact += weight * 0.7
        elif game_status == 'QUESTIONABLE':
            weighted_impact += weight * 0.4
        else:
            weighted_impact += weight * 0.1
    
    # QB status
    qb_status = calculate_qb_injury_status(team_injuries, team)
    
    # O-line health
    oline_health = calculate_oline_health(team_injuries, team)
    
    # Skill position injuries
    skill_injuries = team_injuries[
        team_injuries['position'].str.upper().isin(SKILL_POSITIONS)
    ]
    skill_injury_count = len(skill_injuries)
    skill_out_count = (skill_injuries['game_status'].str.upper() == 'OUT').sum()
    
    # Secondary injuries
    secondary_injuries = team_injuries[
        team_injuries['position'].str.upper().isin(SECONDARY_POSITIONS)
    ]
    secondary_injury_count = len(secondary_injuries)
    secondary_out_count = (secondary_injuries['game_status'].str.upper() == 'OUT').sum()
    
    return {
        'players_out': int(players_out),
        'players_questionable': int(players_questionable),
        'players_doubtful': int(players_doubtful),
        'weighted_injury_impact': float(weighted_impact),
        'qb_status': qb_status,
        'oline_injuries': oline_health['oline_injuries_count'],
        'oline_health_score': oline_health['oline_health_score'],
        'oline_is_compromised': oline_health['oline_is_compromised'],
        'skill_position_injuries': int(skill_injury_count),
        'skill_position_out': int(skill_out_count),
        'secondary_injuries': int(secondary_injury_count),
        'secondary_out': int(secondary_out_count),
    }


def _get_default_injury_features() -> Dict:
    """Return default (no injuries) feature values."""
    return {
        'players_out': 0,
        'players_questionable': 0,
        'players_doubtful': 0,
        'weighted_injury_impact': 0.0,
        'qb_status': 0,
        'oline_injuries': 0,
        'oline_health_score': 100.0,
        'oline_is_compromised': 0,
        'skill_position_injuries': 0,
        'skill_position_out': 0,
        'secondary_injuries': 0,
        'secondary_out': 0,
    }


def add_injury_features_to_games(
    games_df: pd.DataFrame,
    injuries_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add injury features to a games DataFrame.
    
    Args:
        games_df: DataFrame with games (must have: game_id, home_team, away_team, week, season)
        injuries_df: DataFrame with injury data
    
    Returns:
        DataFrame with injury features added
    """
    logger.info("Adding injury features to games...")
    
    # Initialize injury feature columns
    injury_feature_names = [
        'team_players_out', 'team_players_questionable', 'team_weighted_injury_impact',
        'team_qb_status', 'team_oline_injuries', 'team_skill_position_injuries',
        'team_secondary_injuries',
        'opponent_players_out', 'opponent_players_questionable', 'opponent_weighted_injury_impact',
        'opponent_qb_status', 'opponent_oline_injuries', 'opponent_skill_position_injuries',
        'opponent_secondary_injuries',
        'injury_advantage',
    ]
    
    for feature_name in injury_feature_names:
        games_df[feature_name] = np.nan
    
    # Calculate features for each game
    for idx, row in games_df.iterrows():
        try:
            features = calculate_injury_features(
                injuries_df,
                row['home_team'],
                row['away_team'],
                row['week'],
                row['season'],
            )
            
            # Add features to DataFrame
            for feature_name, value in features.items():
                games_df.at[idx, feature_name] = value
            
        except Exception as e:
            logger.warning(f"Error calculating injury features for game {row.get('game_id', 'unknown')}: {e}")
            # Fill with defaults
            for feature_name in injury_feature_names:
                if games_df.at[idx, feature_name] is np.nan:
                    if 'advantage' in feature_name:
                        games_df.at[idx, feature_name] = 0.0
                    elif 'status' in feature_name or 'compromised' in feature_name:
                        games_df.at[idx, feature_name] = 0
                    elif 'score' in feature_name:
                        games_df.at[idx, feature_name] = 100.0
                    else:
                        games_df.at[idx, feature_name] = 0
    
    logger.info(f"Added injury features to {len(games_df)} games")
    return games_df

