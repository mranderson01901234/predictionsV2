"""
NFL TheSportsDB API Integration Module

Fetches NFL team rosters and schedules from TheSportsDB API
and normalizes to standardized formats matching existing modules.

TheSportsDB API Documentation: https://www.thesportsdb.com/api.php
Free tier: Test API key "123" (limited data)
Premium: Requires API key subscription

Module Structure:
- Core API functions: fetch_thesportsdb_schedules(), fetch_thesportsdb_rosters()
- Helper functions: Team normalization, game ID formation, API request handling
- Future extensibility: Module is structured to allow easy addition of:
  * Player stats (fetch_thesportsdb_player_stats())
  * Live scores (fetch_thesportsdb_live_scores())
  * Team information (fetch_thesportsdb_team_info())
  * Historical data (fetch_thesportsdb_historical())
"""

import pandas as pd
import yaml
from pathlib import Path
from typing import List, Optional, Dict
import logging
import time
import re
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import requests (required for API calls)
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests library not available. API fetching will be disabled.")


def load_config() -> dict:
    """Load TheSportsDB configuration."""
    config_path = Path(__file__).parent.parent.parent / "config" / "data" / "thesportsdb.yaml"
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        return {
            "thesportsdb": {
                "api_key": "123",
                "base_url": "https://www.thesportsdb.com/api/v1/json",
                "enabled": True,
            }
        }
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def normalize_team_abbreviation(team: str, season: int = None) -> str:
    """
    Normalize team abbreviations to nflverse standard.
    
    Uses same normalization as schedule.py for consistency.
    """
    team_map = {
        "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BUF": "BUF",
        "CAR": "CAR", "CHI": "CHI", "CIN": "CIN", "CLE": "CLE",
        "DAL": "DAL", "DEN": "DEN", "DET": "DET", "GB": "GB",
        "HOU": "HOU", "IND": "IND", "JAX": "JAX", "KC": "KC",
        "LV": "LV", "LAR": "LAR", "LAC": "LAC", "MIA": "MIA",
        "MIN": "MIN", "NE": "NE", "NO": "NO", "NYG": "NYG",
        "NYJ": "NYJ", "PHI": "PHI", "PIT": "PIT", "SF": "SF",
        "SEA": "SEA", "TB": "TB", "TEN": "TEN", "WAS": "WAS",
        "OAK": "LV", "SD": "LAC", "STL": "LAR",
    }
    
    if team in team_map:
        return team_map[team]
    
    team_upper = team.upper()
    if team_upper in team_map:
        return team_map[team_upper]
    
    logger.warning(f"Unknown team abbreviation: {team}, using as-is")
    return team.upper()[:3] if len(team) >= 3 else team.upper()


def map_thesportsdb_team_name_to_abbreviation(team_name: str) -> Optional[str]:
    """
    Map TheSportsDB team names to normalized abbreviations.
    
    TheSportsDB uses full team names like "Kansas City Chiefs", "Detroit Lions".
    This function maps them to our 3-letter abbreviations.
    """
    # Mapping of TheSportsDB team names to abbreviations
    team_name_map = {
        # AFC
        "Kansas City Chiefs": "KC",
        "Buffalo Bills": "BUF",
        "Miami Dolphins": "MIA",
        "New York Jets": "NYJ",
        "New England Patriots": "NE",
        "Baltimore Ravens": "BAL",
        "Cincinnati Bengals": "CIN",
        "Cleveland Browns": "CLE",
        "Pittsburgh Steelers": "PIT",
        "Houston Texans": "HOU",
        "Indianapolis Colts": "IND",
        "Jacksonville Jaguars": "JAX",
        "Tennessee Titans": "TEN",
        "Denver Broncos": "DEN",
        "Las Vegas Raiders": "LV",
        "Los Angeles Chargers": "LAC",
        "Los Angeles Raiders": "LV",  # Historical
        "Oakland Raiders": "LV",  # Historical
        "San Diego Chargers": "LAC",  # Historical
        
        # NFC
        "Dallas Cowboys": "DAL",
        "New York Giants": "NYG",
        "Philadelphia Eagles": "PHI",
        "Washington Commanders": "WAS",
        "Washington Redskins": "WAS",  # Historical
        "Washington Football Team": "WAS",  # Historical
        "Chicago Bears": "CHI",
        "Detroit Lions": "DET",
        "Green Bay Packers": "GB",
        "Minnesota Vikings": "MIN",
        "Atlanta Falcons": "ATL",
        "Carolina Panthers": "CAR",
        "New Orleans Saints": "NO",
        "Tampa Bay Buccaneers": "TB",
        "Arizona Cardinals": "ARI",
        "Los Angeles Rams": "LAR",
        "San Francisco 49ers": "SF",
        "Seattle Seahawks": "SEA",
        "St. Louis Rams": "LAR",  # Historical
    }
    
    # Try exact match first
    if team_name in team_name_map:
        return team_name_map[team_name]
    
    # Try case-insensitive match
    team_name_lower = team_name.lower()
    for key, abbrev in team_name_map.items():
        if key.lower() == team_name_lower:
            return abbrev
    
    # Try partial match (e.g., "Chiefs" -> "KC")
    for key, abbrev in team_name_map.items():
        if team_name_lower in key.lower() or key.lower() in team_name_lower:
            logger.debug(f"Partial match: '{team_name}' -> '{abbrev}'")
            return abbrev
    
    logger.warning(f"Could not map team name '{team_name}' to abbreviation")
    return None


def form_game_id(season: int, week: int, away_team: str, home_team: str) -> str:
    """Form game_id matching schedule.py format."""
    away_norm = normalize_team_abbreviation(away_team)
    home_norm = normalize_team_abbreviation(home_team)
    return f"nfl_{season}_{week:02d}_{away_norm}_{home_norm}"


def _make_api_request(url: str, params: Optional[Dict] = None, max_retries: int = 3) -> Optional[Dict]:
    """
    Make API request with retry logic, error handling, and rate limiting.
    
    Implements exponential backoff for rate limits and network errors.
    Respects rate limits from config.
    
    Args:
        url: API endpoint URL
        params: Optional query parameters
        max_retries: Maximum number of retry attempts
    
    Returns:
        JSON response as dict, or None if failed
    """
    if not REQUESTS_AVAILABLE:
        logger.error("requests library not available. Cannot make API calls.")
        return None
    
    config = load_config()
    if not config.get("thesportsdb", {}).get("enabled", True):
        logger.warning("TheSportsDB API is disabled in config")
        return None
    
    # Get rate limit settings from config
    rate_limit_config = config.get("thesportsdb", {}).get("rate_limit", {})
    requests_per_minute = rate_limit_config.get("requests_per_minute", 10)
    min_delay = 60.0 / requests_per_minute  # Minimum delay between requests
    
    for attempt in range(max_retries):
        try:
            # Rate limiting: ensure minimum delay between requests
            if attempt > 0:
                time.sleep(min_delay)
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                # Check for API error messages
                if "events" in data and data["events"] is None:
                    logger.warning(f"API returned null events for URL: {url}")
                    return None
                return data
            elif response.status_code == 401:
                logger.error("Invalid API key. Check your TheSportsDB API key in config.")
                return None
            elif response.status_code == 429:
                logger.warning(f"Rate limit exceeded. Waiting before retry...")
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                logger.warning(f"API request failed with status {response.status_code}: {response.text[:200]}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return None
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None
    
    return None


def fetch_nfl_teams() -> List[Dict]:
    """
    Fetch all NFL teams from TheSportsDB.
    
    Returns:
        List of team dictionaries with team_id, name, etc.
    """
    config = load_config()
    api_key = config.get("thesportsdb", {}).get("api_key", "123")
    base_url = config.get("thesportsdb", {}).get("base_url", "https://www.thesportsdb.com/api/v1/json")
    league_id = config.get("thesportsdb", {}).get("nfl", {}).get("league_id", "4391")
    
    url = f"{base_url}/{api_key}/lookup_all_teams.php"
    params = {"id": league_id}
    
    logger.info(f"Fetching NFL teams from TheSportsDB (league_id: {league_id})")
    data = _make_api_request(url, params)
    
    if data is None or "teams" not in data:
        logger.error("Failed to fetch NFL teams from TheSportsDB")
        return []
    
    teams = data.get("teams", [])
    logger.info(f"Fetched {len(teams)} NFL teams")
    return teams


def fetch_thesportsdb_rosters(team_id: str) -> pd.DataFrame:
    """
    Fetch NFL team roster from TheSportsDB API.
    
    Args:
        team_id: TheSportsDB team ID (e.g., "133602" for Kansas City Chiefs)
    
    Returns:
        DataFrame with standardized roster columns matching rosters.py format:
        - team_id: TheSportsDB team ID
        - team: Team abbreviation (normalized to nflverse standard)
        - team_name: Full team name
        - player_id: TheSportsDB player ID
        - player_name: Player full name
        - position: Player position
        - jersey_number: Jersey number (if available)
        - nationality: Player nationality (if available)
        - date_born: Date of birth (if available)
        - height: Height in cm (if available)
        - weight: Weight in kg (if available)
        - thumb: Player photo URL (if available)
    
    Raises:
        ValueError: If team_id is invalid or empty
    """
    if not team_id or not str(team_id).strip():
        raise ValueError("team_id cannot be empty")
    
    config = load_config()
    api_key = config.get("thesportsdb", {}).get("api_key", "123")
    base_url = config.get("thesportsdb", {}).get("base_url", "https://www.thesportsdb.com/api/v1/json")
    
    url = f"{base_url}/{api_key}/lookup_all_players.php"
    params = {"id": team_id}
    
    logger.info(f"Fetching roster for team_id: {team_id}")
    data = _make_api_request(url, params)
    
    if data is None:
        logger.error(f"Failed to fetch roster for team_id: {team_id}")
        return pd.DataFrame()
    
    players = data.get("player", [])
    if not players:
        logger.warning(f"No players found for team_id: {team_id}")
        return pd.DataFrame()
    
    # Get team name for normalization
    team_name = None
    if players:
        team_name = players[0].get("strTeam", "")
    
    # Normalize team name to abbreviation
    team_abbrev = map_thesportsdb_team_name_to_abbreviation(team_name) if team_name else None
    
    # Build roster DataFrame
    roster_data = []
    for player in players:
        roster_data.append({
            "team_id": team_id,
            "team": team_abbrev,
            "team_name": team_name,
            "player_id": player.get("idPlayer", ""),
            "player_name": player.get("strPlayer", ""),
            "position": player.get("strPosition", ""),
            "jersey_number": player.get("strNumber", ""),
            "nationality": player.get("strNationality", ""),
            "date_born": player.get("dateBorn", ""),
            "height": player.get("strHeight", ""),
            "weight": player.get("strWeight", ""),
            "thumb": player.get("strThumb", ""),  # Player photo URL
        })
    
    df = pd.DataFrame(roster_data)
    
    # Clean up data types
    if "jersey_number" in df.columns:
        df["jersey_number"] = pd.to_numeric(df["jersey_number"], errors="coerce")
    
    # Parse date_born
    if "date_born" in df.columns:
        df["date_born"] = pd.to_datetime(df["date_born"], errors="coerce", format="%Y-%m-%d")
    
    logger.info(f"Fetched {len(df)} players for team {team_abbrev or team_name}")
    return df


def _parse_thesportsdb_event(event: Dict, season: int) -> Optional[Dict]:
    """
    Parse a single event/game from TheSportsDB API response.
    
    Args:
        event: Event dictionary from API
        season: Season year for context
    
    Returns:
        Dictionary with parsed game data, or None if invalid
    """
    try:
        # Extract date
        date_str = event.get("dateEvent", "")
        if not date_str:
            return None
        
        date = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(date):
            return None
        
        # Extract teams
        home_team_name = event.get("strHomeTeam", "")
        away_team_name = event.get("strAwayTeam", "")
        
        if not home_team_name or not away_team_name:
            return None
        
        # Map to abbreviations
        home_team = map_thesportsdb_team_name_to_abbreviation(home_team_name)
        away_team = map_thesportsdb_team_name_to_abbreviation(away_team_name)
        
        if not home_team or not away_team:
            logger.debug(f"Could not map teams: {home_team_name} / {away_team_name}")
            return None
        
        # Extract scores
        home_score_str = event.get("intHomeScore", "")
        away_score_str = event.get("intAwayScore", "")
        
        home_score = pd.to_numeric(home_score_str, errors="coerce")
        away_score = pd.to_numeric(away_score_str, errors="coerce")
        
        # If scores are missing, set to 0 (future games)
        if pd.isna(home_score):
            home_score = 0
        if pd.isna(away_score):
            away_score = 0
        
        # Determine week (approximate - TheSportsDB may not have week info)
        # We'll need to infer from date or leave as None
        week = None
        
        # Try to extract week from event name or other fields
        event_name = event.get("strEvent", "")
        if "Week" in event_name or "week" in event_name:
            # Try to extract week number from event name
            week_match = re.search(r'[Ww]eek\s+(\d+)', event_name)
            if week_match:
                week = int(week_match.group(1))
        
        return {
            "date": date,
            "home_team": home_team,
            "away_team": away_team,
            "home_team_name": home_team_name,
            "away_team_name": away_team_name,
            "home_score": int(home_score),
            "away_score": int(away_score),
            "season": season,
            "week": week,
            "event_id": event.get("idEvent", ""),
            "venue": event.get("strVenue", ""),
        }
    except Exception as e:
        logger.warning(f"Error parsing event: {e}")
        return None


def fetch_thesportsdb_schedules(seasons: List[int]) -> pd.DataFrame:
    """
    Fetch NFL schedules from TheSportsDB API.
    
    Args:
        seasons: List of season years to fetch (e.g., [2023, 2024])
    
    Returns:
        DataFrame with standardized schedule columns matching schedule.py format:
        - game_id: Standardized game ID (format: nfl_{season}_{week:02d}_{away}_{home})
        - season: Season year
        - week: Week number (if available, else None)
        - date: Game date (datetime)
        - home_team: Home team abbreviation (normalized to nflverse standard)
        - away_team: Away team abbreviation (normalized to nflverse standard)
        - home_score: Home team score (0 if not played yet)
        - away_score: Away team score (0 if not played yet)
        - venue: Venue name (if available)
    
    Note:
        - Week numbers may be inferred from event names or approximated from dates
        - Scores are 0 for future games
        - Team abbreviations are normalized to match schedule.py format
    """
    config = load_config()
    api_key = config.get("thesportsdb", {}).get("api_key", "123")
    base_url = config.get("thesportsdb", {}).get("base_url", "https://www.thesportsdb.com/api/v1/json")
    league_id = config.get("thesportsdb", {}).get("nfl", {}).get("league_id", "4391")
    
    logger.info(f"Fetching NFL schedules from TheSportsDB for seasons: {seasons}")
    
    all_games = []
    
    for season in seasons:
        # TheSportsDB uses season format like "2023-2024" or just year
        # Try both formats
        url = f"{base_url}/{api_key}/eventsseason.php"
        params = {"id": league_id, "s": str(season)}
        
        logger.info(f"Fetching schedule for season {season}")
        data = _make_api_request(url, params)
        
        if data is None:
            logger.warning(f"Failed to fetch schedule for season {season}")
            continue
        
        events = data.get("events", [])
        if not events:
            logger.warning(f"No events found for season {season}")
            continue
        
        # Parse events
        parsed_games = []
        for event in events:
            parsed = _parse_thesportsdb_event(event, season)
            if parsed:
                parsed_games.append(parsed)
        
        logger.info(f"Parsed {len(parsed_games)} games for season {season}")
        all_games.extend(parsed_games)
        
        # Rate limiting - be respectful
        time.sleep(0.5)
    
    if not all_games:
        logger.warning("No games found in any season")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_games)
    
    # Form game_id (need week - if missing, we'll use date-based approximation)
    # For games without week, we'll try to infer from date or leave as None
    def _form_game_id_with_week(row):
        if row["week"] is not None:
            return form_game_id(row["season"], row["week"], row["away_team"], row["home_team"])
        else:
            # Fallback: use date-based week approximation
            # NFL season typically starts in September
            date = row["date"]
            if pd.isna(date):
                return None
            
            # Approximate week from date (rough estimate)
            # This is not perfect but better than nothing
            season_start = pd.Timestamp(f"{row['season']}-09-01")
            days_diff = (date - season_start).days
            approx_week = max(1, min(18, (days_diff // 7) + 1))
            return form_game_id(row["season"], approx_week, row["away_team"], row["home_team"])
    
    df["game_id"] = df.apply(_form_game_id_with_week, axis=1)
    
    # Select and order columns to match schedule.py format
    schedule_df = pd.DataFrame({
        "game_id": df["game_id"],
        "season": df["season"],
        "week": df["week"],  # May be None
        "date": df["date"],
        "home_team": df["home_team"],
        "away_team": df["away_team"],
        "home_score": df["home_score"],
        "away_score": df["away_score"],
    })
    
    # Add optional columns if available
    if "venue" in df.columns:
        schedule_df["venue"] = df["venue"]
    
    # Remove rows with missing game_id
    initial_len = len(schedule_df)
    schedule_df = schedule_df[schedule_df["game_id"].notna()]
    if len(schedule_df) < initial_len:
        logger.warning(f"Dropped {initial_len - len(schedule_df)} rows with missing game_id")
    
    # Sort by season, date
    schedule_df = schedule_df.sort_values(["season", "date"]).reset_index(drop=True)
    
    logger.info(f"Fetched {len(schedule_df)} games total across {len(seasons)} seasons")
    return schedule_df


# ============================================================================
# Future Extensibility Functions (Template)
# ============================================================================
# These functions can be added as needed for additional TheSportsDB features

# def fetch_thesportsdb_player_stats(player_id: str, season: Optional[int] = None) -> pd.DataFrame:
#     """
#     Fetch player statistics from TheSportsDB API.
#     
#     Future addition: Player-level stats (passing, rushing, receiving, etc.)
#     """
#     pass

# def fetch_thesportsdb_live_scores() -> pd.DataFrame:
#     """
#     Fetch live NFL scores from TheSportsDB API.
#     
#     Future addition: Real-time game scores and updates
#     """
#     pass

# def fetch_thesportsdb_team_info(team_id: str) -> Dict:
#     """
#     Fetch detailed team information from TheSportsDB API.
#     
#     Future addition: Team details, stadium info, etc.
#     """
#     pass


if __name__ == "__main__":
    # Test functions
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test-rosters":
        # Test roster fetching
        print("Testing roster fetching...")
        teams = fetch_nfl_teams()
        if teams:
            # Use first team for testing
            test_team_id = teams[0].get("idTeam", "")
            print(f"\nFetching roster for team: {teams[0].get('strTeam', 'Unknown')} (ID: {test_team_id})")
            try:
                roster_df = fetch_thesportsdb_rosters(test_team_id)
                print(f"\nFetched {len(roster_df)} players")
                if len(roster_df) > 0:
                    print("\nFirst 5 players:")
                    print(roster_df[["player_name", "position", "jersey_number", "team"]].head())
            except Exception as e:
                print(f"Error fetching roster: {e}")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "--test-schedules":
        # Test schedule fetching
        print("Testing schedule fetching...")
        test_seasons = [2023, 2024]
        schedule_df = fetch_thesportsdb_schedules(test_seasons)
        print(f"\nFetched {len(schedule_df)} games")
        if len(schedule_df) > 0:
            print("\nFirst 5 games:")
            print(schedule_df.head())
    
    else:
        # Default: test both
        print("Testing TheSportsDB API integration...")
        print("\n1. Testing team fetching...")
        teams = fetch_nfl_teams()
        print(f"Found {len(teams)} teams")
        if teams:
            print(f"Sample team: {teams[0].get('strTeam', 'Unknown')} (ID: {teams[0].get('idTeam', '')})")
        
        print("\n2. Testing schedule fetching...")
        schedule_df = fetch_thesportsdb_schedules([2023])
        print(f"Fetched {len(schedule_df)} games")
        if len(schedule_df) > 0:
            print("\nSample games:")
            print(schedule_df[["game_id", "season", "date", "home_team", "away_team", "home_score", "away_score"]].head())

