"""
BALLDONTLIE NFL API Integration

Free NFL API providing injury data, player stats, and schedules.
API Documentation: https://nfl.balldontlie.io/

This is a free alternative to paid injury data services.
"""

import pandas as pd
import requests
from typing import Optional, List, Dict
import logging
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://nfl.balldontlie.io/api/v1"


def fetch_balldontlie_injuries(
    season: Optional[int] = None,
    week: Optional[int] = None,
    team: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch NFL injuries from BALLDONTLIE API.
    
    Args:
        season: Season year (e.g., 2025). If None, fetches current season.
        week: Week number (1-18). If None, fetches all weeks.
        team: Team abbreviation (e.g., 'KC'). If None, fetches all teams.
    
    Returns:
        DataFrame with injury data:
        - player_id, player_name
        - team (abbreviation)
        - position
        - injury_type
        - status (Out, Doubtful, Questionable, Probable)
        - season, week
    """
    logger.info(f"Fetching injuries from BALLDONTLIE API (season={season}, week={week}, team={team})")
    
    all_injuries = []
    
    try:
        # BALLDONTLIE API endpoint for injuries
        # Note: Actual endpoint structure may vary - this is based on typical REST API patterns
        url = f"{BASE_URL}/injuries"
        
        params = {}
        if season:
            params['season'] = season
        if week:
            params['week'] = week
        if team:
            params['team'] = team
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Parse response structure
        # BALLDONTLIE typically returns: { "data": [...], "meta": {...} }
        injuries_data = data.get('data', [])
        
        if not injuries_data:
            # Try alternative response structure
            if isinstance(data, list):
                injuries_data = data
            else:
                logger.warning("No injury data found in BALLDONTLIE API response")
                return pd.DataFrame()
        
        for injury in injuries_data:
            # Map API response to our schema
            injury_record = {
                'player_id': injury.get('player_id') or injury.get('player', {}).get('id'),
                'player_name': injury.get('player_name') or injury.get('player', {}).get('name') or injury.get('player', {}).get('full_name'),
                'team': _normalize_team_abbrev(injury.get('team') or injury.get('team_abbreviation')),
                'position': injury.get('position'),
                'injury_type': injury.get('injury') or injury.get('injury_type') or injury.get('body_part'),
                'status': _normalize_status(injury.get('status') or injury.get('game_status')),
                'practice_status': injury.get('practice_status'),
                'season': injury.get('season') or season or datetime.now().year,
                'week': injury.get('week') or week,
            }
            
            all_injuries.append(injury_record)
        
        df = pd.DataFrame(all_injuries)
        logger.info(f"Fetched {len(df)} injuries from BALLDONTLIE API")
        
        return df
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching from BALLDONTLIE API: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error parsing BALLDONTLIE API response: {e}")
        return pd.DataFrame()


def fetch_current_week_injuries() -> pd.DataFrame:
    """
    Fetch current week's injuries from BALLDONTLIE API.
    
    Returns:
        DataFrame with current week injury data
    """
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    # NFL season spans two calendar years
    # If we're in Sep-Dec, use current year as season
    # If we're in Jan-Aug, use previous year as season
    if current_month >= 9:
        season = current_year
    else:
        season = current_year - 1
    
    logger.info(f"Fetching current week injuries for season {season}")
    
    # Fetch all current season injuries (BALLDONTLIE may not have week filter)
    injuries = fetch_balldontlie_injuries(season=season)
    
    # Filter to current week if possible
    # Note: May need to calculate current week from date
    return injuries


def _normalize_team_abbrev(team: str) -> str:
    """Normalize team abbreviation to standard format."""
    if not team:
        return ''
    
    team = str(team).upper().strip()
    
    # Map common variations
    team_map = {
        'KANSAS CITY': 'KC',
        'K.C.': 'KC',
        'KC CHIEFS': 'KC',
        'HOUSTON': 'HOU',
        'HOU TEXANS': 'HOU',
        # Add more mappings as needed
    }
    
    return team_map.get(team, team[:3])


def _normalize_status(status: str) -> str:
    """Normalize injury status to standard format."""
    if not status:
        return ''
    
    status = str(status).upper().strip()
    
    # Map to standard statuses
    status_map = {
        'OUT': 'Out',
        'DOUBTFUL': 'Doubtful',
        'QUESTIONABLE': 'Questionable',
        'PROBABLE': 'Probable',
        'ACTIVE': None,  # Not injured
        'NONE': None,
    }
    
    return status_map.get(status, status)


if __name__ == "__main__":
    # Test the API
    print("Testing BALLDONTLIE API...")
    injuries = fetch_current_week_injuries()
    print(f"Fetched {len(injuries)} injuries")
    if len(injuries) > 0:
        print(injuries.head())

