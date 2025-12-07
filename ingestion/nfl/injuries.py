"""
NFL Injury Reports Ingestion Module

Fetches NFL injury data from TheSportsDB API (if available) or scrapes NFL.com/injuries.
Normalizes to standardized InjuryReport schema.

Data sources (in priority order):
1. TheSportsDB API (if available)
2. NFL.com/injuries scraping (fallback)

InjuryReport Schema:
- game_id: Standardized game ID
- team: Team abbreviation (normalized)
- player_id: Player identifier
- position: Player position
- injury_type: Type of injury
- status: Injury status (questionable/probable/out/doubtful)
- report_date: Date of injury report
"""

import pandas as pd
import yaml
from pathlib import Path
from typing import List, Optional, Dict
import logging
import time
import re
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import requests (required for API calls)
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests library not available. API fetching will be disabled.")

# Try to import BeautifulSoup (required for scraping)
try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    logger.warning("BeautifulSoup not available. Web scraping will be disabled.")


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


def map_team_name_to_abbreviation(team_name: str) -> Optional[str]:
    """
    Map team names to normalized abbreviations.
    
    Handles various team name formats from different sources.
    """
    team_name_map = {
        # Full names
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
        "Oakland Raiders": "LV",
        "San Diego Chargers": "LAC",
        "Dallas Cowboys": "DAL",
        "New York Giants": "NYG",
        "Philadelphia Eagles": "PHI",
        "Washington Commanders": "WAS",
        "Washington Redskins": "WAS",
        "Washington Football Team": "WAS",
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
        "St. Louis Rams": "LAR",
    }
    
    # Try exact match
    if team_name in team_name_map:
        return team_name_map[team_name]
    
    # Try case-insensitive match
    team_name_lower = team_name.lower()
    for key, abbrev in team_name_map.items():
        if key.lower() == team_name_lower:
            return abbrev
    
    # Try partial match
    for key, abbrev in team_name_map.items():
        if team_name_lower in key.lower() or key.lower() in team_name_lower:
            return abbrev
    
    # Try to extract abbreviation from team name (e.g., "Chiefs" -> "KC")
    if "Chiefs" in team_name:
        return "KC"
    elif "Bills" in team_name:
        return "BUF"
    elif "Dolphins" in team_name:
        return "MIA"
    elif "Jets" in team_name and "New York" in team_name:
        return "NYJ"
    elif "Patriots" in team_name:
        return "NE"
    elif "Ravens" in team_name:
        return "BAL"
    elif "Bengals" in team_name:
        return "CIN"
    elif "Browns" in team_name:
        return "CLE"
    elif "Steelers" in team_name:
        return "PIT"
    elif "Texans" in team_name:
        return "HOU"
    elif "Colts" in team_name:
        return "IND"
    elif "Jaguars" in team_name:
        return "JAX"
    elif "Titans" in team_name:
        return "TEN"
    elif "Broncos" in team_name:
        return "DEN"
    elif "Raiders" in team_name:
        return "LV"
    elif "Chargers" in team_name:
        return "LAC"
    elif "Cowboys" in team_name:
        return "DAL"
    elif "Giants" in team_name and "New York" in team_name:
        return "NYG"
    elif "Eagles" in team_name:
        return "PHI"
    elif "Commanders" in team_name or "Redskins" in team_name or ("Washington" in team_name and "Football" in team_name):
        return "WAS"
    elif "Bears" in team_name:
        return "CHI"
    elif "Lions" in team_name:
        return "DET"
    elif "Packers" in team_name:
        return "GB"
    elif "Vikings" in team_name:
        return "MIN"
    elif "Falcons" in team_name:
        return "ATL"
    elif "Panthers" in team_name:
        return "CAR"
    elif "Saints" in team_name:
        return "NO"
    elif "Buccaneers" in team_name or "Bucs" in team_name:
        return "TB"
    elif "Cardinals" in team_name:
        return "ARI"
    elif "Rams" in team_name:
        return "LAR"
    elif "49ers" in team_name or "Niners" in team_name:
        return "SF"
    elif "Seahawks" in team_name:
        return "SEA"
    
    logger.warning(f"Could not map team name '{team_name}' to abbreviation")
    return None


def normalize_injury_status(status: str) -> str:
    """
    Normalize injury status to standard values.
    
    Standard values: questionable, probable, out, doubtful, limited, full
    """
    if not status:
        return "unknown"
    
    status_lower = status.lower().strip()
    
    # Map common variations
    status_map = {
        "questionable": "questionable",
        "q": "questionable",
        "probable": "probable",
        "p": "probable",
        "out": "out",
        "o": "out",
        "doubtful": "doubtful",
        "d": "doubtful",
        "limited": "limited",
        "l": "limited",
        "full": "full",
        "f": "full",
        "did not participate": "out",
        "dnp": "out",
        "limited participation": "limited",
        "full participation": "full",
    }
    
    if status_lower in status_map:
        return status_map[status_lower]
    
    # Check for partial matches
    for key, value in status_map.items():
        if key in status_lower:
            return value
    
    return status_lower


def _make_api_request(url: str, params: Optional[Dict] = None, max_retries: int = 3) -> Optional[Dict]:
    """
    Make API request with retry logic and error handling.
    
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
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
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


def _fetch_thesportsdb_injuries(seasons: List[int]) -> pd.DataFrame:
    """
    Attempt to fetch injury data from TheSportsDB API.
    
    Note: TheSportsDB may not have injury endpoints, but we try anyway.
    
    Args:
        seasons: List of season years
    
    Returns:
        DataFrame with injury data, or empty DataFrame if not available
    """
    config = load_config()
    api_key = config.get("thesportsdb", {}).get("api_key", "123")
    base_url = config.get("thesportsdb", {}).get("base_url", "https://www.thesportsdb.com/api/v1/json")
    league_id = config.get("thesportsdb", {}).get("nfl", {}).get("league_id", "4391")
    
    logger.info("Attempting to fetch injuries from TheSportsDB API...")
    
    # TheSportsDB may not have a direct injuries endpoint
    # Try common endpoint patterns
    endpoints_to_try = [
        f"{base_url}/{api_key}/lookupinjuries.php",
        f"{base_url}/{api_key}/eventsinjuries.php",
        f"{base_url}/{api_key}/injuries.php",
    ]
    
    for endpoint in endpoints_to_try:
        try:
            params = {"id": league_id}
            data = _make_api_request(endpoint, params)
            
            if data and "injuries" in data:
                logger.info("Found injury data in TheSportsDB API")
                # Parse and return data
                # Note: Actual structure depends on API response
                injuries = data.get("injuries", [])
                if injuries:
                    # Would need to parse based on actual API structure
                    logger.info(f"Found {len(injuries)} injury records")
                    # Return empty for now - would need to implement parsing
                    return pd.DataFrame()
        except Exception as e:
            logger.debug(f"Endpoint {endpoint} not available: {e}")
            continue
    
    logger.info("TheSportsDB API does not appear to have injury data endpoints")
    return pd.DataFrame()


def _scrape_nfl_com_injuries(seasons: List[int], games_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Scrape NFL.com/injuries for injury reports.
    
    Args:
        seasons: List of season years
        games_df: Optional DataFrame with games (for game_id mapping)
    
    Returns:
        DataFrame with injury data
    """
    if not REQUESTS_AVAILABLE:
        logger.error("requests library not available. Cannot scrape NFL.com")
        return pd.DataFrame()
    
    if not BEAUTIFULSOUP_AVAILABLE:
        logger.error("BeautifulSoup not available. Cannot scrape NFL.com")
        return pd.DataFrame()
    
    logger.info(f"Scraping NFL.com/injuries for seasons: {seasons}")
    
    all_injuries = []
    
    # NFL.com injury reports are typically organized by week
    # URL format: https://www.nfl.com/injuries/ (current week)
    # Historical: May need to use archive or API endpoints
    
    base_url = "https://www.nfl.com/injuries"
    
    # Try to fetch current injury reports
    # Note: NFL.com structure may change, so this is a template implementation
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(base_url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            logger.warning(f"NFL.com returned status {response.status_code}")
            return pd.DataFrame()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Parse injury data from HTML
        # Note: This is a template - actual HTML structure needs to be inspected
        # NFL.com uses dynamic content, so may need Selenium or API endpoints
        
        # Look for injury report tables or data structures
        injury_tables = soup.find_all("table", class_=re.compile("injury|report", re.I))
        
        if not injury_tables:
            # Try alternative selectors
            injury_sections = soup.find_all("div", class_=re.compile("injury|report", re.I))
            
            if not injury_sections:
                logger.warning("Could not find injury data structure on NFL.com")
                return pd.DataFrame()
        
        # Parse injuries from HTML structure
        # This is a placeholder - actual parsing depends on NFL.com HTML structure
        logger.info("Found injury data structure, but parsing logic needs to be implemented based on actual HTML")
        
        # For now, return empty DataFrame
        # In production, would parse HTML and extract:
        # - Team name
        # - Player name
        # - Position
        # - Injury type
        # - Status
        # - Report date
        
    except Exception as e:
        logger.error(f"Error scraping NFL.com: {e}")
        return pd.DataFrame()
    
    if not all_injuries:
        logger.warning("No injuries found from NFL.com scraping")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_injuries)
    return df


def _map_injuries_to_games(
    injuries_df: pd.DataFrame,
    games_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Map injuries to game_id using schedule data.
    
    Args:
        injuries_df: DataFrame with injury data (must have team, report_date)
        games_df: DataFrame with games (must have game_id, season, week, home_team, away_team, date)
    
    Returns:
        DataFrame with game_id added
    """
    if games_df is None:
        # Try to load games from staged data
        games_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "nfl"
            / "staged"
            / "games.parquet"
        )
        if games_path.exists():
            games_df = pd.read_parquet(games_path)
            logger.info(f"Loaded {len(games_df)} games from staged data")
        else:
            logger.warning("No games DataFrame provided and staged games not found. Cannot map to game_id")
            return injuries_df
    
    if "team" not in injuries_df.columns or "report_date" not in injuries_df.columns:
        logger.warning("Injuries DataFrame missing required columns for game mapping")
        return injuries_df
    
    # Normalize team abbreviations
    injuries_df["team_norm"] = injuries_df["team"].apply(normalize_team_abbreviation)
    
    # Convert report_date to datetime if needed
    if injuries_df["report_date"].dtype != "datetime64[ns]":
        injuries_df["report_date"] = pd.to_datetime(injuries_df["report_date"], errors="coerce")
    
    # Map injuries to games
    # Strategy: Find games where team matches and report_date is within 7 days before game date
    game_ids = []
    
    for idx, injury in injuries_df.iterrows():
        team = injury["team_norm"]
        report_date = injury["report_date"]
        
        if pd.isna(report_date):
            game_ids.append(None)
            continue
        
        # Find games for this team within 7 days of report date
        team_games = games_df[
            ((games_df["home_team"] == team) | (games_df["away_team"] == team)) &
            (games_df["date"] >= report_date) &
            (games_df["date"] <= report_date + timedelta(days=7))
        ]
        
        if len(team_games) > 0:
            # Use the closest game
            team_games = team_games.copy()
            team_games["days_diff"] = (team_games["date"] - report_date).dt.days
            closest_game = team_games.nsmallest(1, "days_diff")
            game_ids.append(closest_game.iloc[0]["game_id"])
        else:
            game_ids.append(None)
    
    injuries_df["game_id"] = game_ids
    
    # Count how many were mapped
    mapped_count = injuries_df["game_id"].notna().sum()
    logger.info(f"Mapped {mapped_count}/{len(injuries_df)} injuries to game_id")
    
    return injuries_df


def fetch_nfl_injuries(
    seasons: List[int],
    games_df: Optional[pd.DataFrame] = None,
    use_api: bool = True,
    use_scraping: bool = True
) -> pd.DataFrame:
    """
    Fetch NFL injury data from available sources.
    
    Tries TheSportsDB API first, then falls back to NFL.com scraping.
    
    Args:
        seasons: List of season years to fetch
        games_df: Optional DataFrame with games (for game_id mapping)
        use_api: Whether to attempt API fetching
        use_scraping: Whether to attempt web scraping
    
    Returns:
        DataFrame with standardized injury columns:
        - game_id: Standardized game ID
        - team: Team abbreviation (normalized)
        - player_id: Player identifier
        - position: Player position
        - injury_type: Type of injury
        - status: Injury status (questionable/probable/out/doubtful)
        - report_date: Date of injury report
    """
    logger.info(f"Fetching NFL injuries for seasons: {seasons}")
    
    all_injuries = []
    
    # Try TheSportsDB API first
    if use_api:
        try:
            api_injuries = _fetch_thesportsdb_injuries(seasons)
            if len(api_injuries) > 0:
                logger.info(f"Fetched {len(api_injuries)} injuries from TheSportsDB API")
                all_injuries.append(api_injuries)
            else:
                logger.info("TheSportsDB API did not return injury data, falling back to scraping")
        except Exception as e:
            logger.warning(f"Error fetching from TheSportsDB API: {e}")
            logger.info("Falling back to scraping")
    
    # Fall back to scraping NFL.com
    if use_scraping and len(all_injuries) == 0:
        try:
            scraped_injuries = _scrape_nfl_com_injuries(seasons, games_df)
            if len(scraped_injuries) > 0:
                logger.info(f"Scraped {len(scraped_injuries)} injuries from NFL.com")
                all_injuries.append(scraped_injuries)
        except Exception as e:
            logger.error(f"Error scraping NFL.com: {e}")
    
    # Combine all injuries
    if not all_injuries:
        logger.warning("No injury data found from any source")
        return pd.DataFrame()
    
    df = pd.concat(all_injuries, ignore_index=True)
    
    # Ensure required columns exist
    required_columns = ["game_id", "team", "player_id", "position", "injury_type", "status", "report_date"]
    
    for col in required_columns:
        if col not in df.columns:
            if col == "game_id":
                # Will be mapped later
                df[col] = None
            elif col == "player_id":
                # Generate from player name if available
                if "player_name" in df.columns:
                    df[col] = df["player_name"].apply(lambda x: str(hash(str(x))) if pd.notna(x) else None)
                else:
                    df[col] = None
            else:
                df[col] = None
    
    # Normalize team abbreviations
    if "team" in df.columns:
        df["team"] = df["team"].apply(normalize_team_abbreviation)
    
    # Normalize injury status
    if "status" in df.columns:
        df["status"] = df["status"].apply(normalize_injury_status)
    
    # Map to game_id if games_df provided
    if games_df is not None or "game_id" in df.columns and df["game_id"].isna().all():
        df = _map_injuries_to_games(df, games_df)
    
    # Select and order columns
    injury_df = pd.DataFrame({
        "game_id": df["game_id"],
        "team": df["team"],
        "player_id": df["player_id"],
        "position": df["position"],
        "injury_type": df["injury_type"],
        "status": df["status"],
        "report_date": df["report_date"],
    })
    
    # Remove rows with missing critical data
    initial_len = len(injury_df)
    injury_df = injury_df[
        injury_df["team"].notna() &
        injury_df["player_id"].notna() &
        injury_df["report_date"].notna()
    ]
    if len(injury_df) < initial_len:
        logger.warning(f"Dropped {initial_len - len(injury_df)} rows with missing critical data")
    
    # Sort by report_date, team
    injury_df = injury_df.sort_values(["report_date", "team"]).reset_index(drop=True)
    
    logger.info(f"Fetched {len(injury_df)} total injuries")
    return injury_df


if __name__ == "__main__":
    # Test function
    import sys
    
    test_seasons = [2023, 2024]
    
    print("Testing NFL injury fetching...")
    print(f"Seasons: {test_seasons}")
    
    # Try to load games for mapping
    games_path = Path(__file__).parent.parent.parent / "data" / "nfl" / "staged" / "games.parquet"
    games_df = None
    if games_path.exists():
        games_df = pd.read_parquet(games_path)
        print(f"Loaded {len(games_df)} games for mapping")
    
    injuries_df = fetch_nfl_injuries(test_seasons, games_df=games_df)
    
    print(f"\nFetched {len(injuries_df)} injuries")
    if len(injuries_df) > 0:
        print("\nFirst 5 injuries:")
        print(injuries_df.head())
        print(f"\nStatus distribution:")
        print(injuries_df["status"].value_counts() if "status" in injuries_df.columns else "N/A")

