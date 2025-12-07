"""
NFL Odds Ingestion Module

Fetches historical NFL betting odds (spreads, totals) from free sources
and normalizes to the MarketSnapshot schema.

Data sources (in priority order):
1. The Odds API (free tier: 500 requests/month)
2. CSV file (manual fallback)
3. nflverse schedule data (last resort)
"""

import pandas as pd
import yaml
from pathlib import Path
from typing import List, Optional, Dict
import logging
from datetime import datetime, timedelta
import time

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
    """Load NFL data configuration."""
    config_path = Path(__file__).parent.parent.parent / "config" / "data" / "nfl.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def normalize_team_abbreviation(team: str) -> str:
    """
    Normalize team abbreviations to match schedule module.
    
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


def form_game_id(season: int, week: int, away_team: str, home_team: str) -> str:
    """Form game_id matching schedule.py format."""
    away_norm = normalize_team_abbreviation(away_team)
    home_norm = normalize_team_abbreviation(home_team)
    return f"nfl_{season}_{week:02d}_{away_norm}_{home_norm}"


def map_odds_api_team_to_abbreviation(team_name: str) -> Optional[str]:
    """
    Map The Odds API team names to our normalized abbreviations.
    
    The Odds API uses full team names like "Kansas City Chiefs", "Detroit Lions".
    We need to map these to our 3-letter abbreviations.
    
    Args:
        team_name: Full team name from The Odds API
    
    Returns:
        Normalized team abbreviation or None if not found
    """
    # Mapping of The Odds API team names to our abbreviations
    team_mapping = {
        "Arizona Cardinals": "ARI",
        "Atlanta Falcons": "ATL",
        "Baltimore Ravens": "BAL",
        "Buffalo Bills": "BUF",
        "Carolina Panthers": "CAR",
        "Chicago Bears": "CHI",
        "Cincinnati Bengals": "CIN",
        "Cleveland Browns": "CLE",
        "Dallas Cowboys": "DAL",
        "Denver Broncos": "DEN",
        "Detroit Lions": "DET",
        "Green Bay Packers": "GB",
        "Houston Texans": "HOU",
        "Indianapolis Colts": "IND",
        "Jacksonville Jaguars": "JAX",
        "Kansas City Chiefs": "KC",
        "Las Vegas Raiders": "LV",
        "Los Angeles Rams": "LAR",
        "Los Angeles Chargers": "LAC",
        "Miami Dolphins": "MIA",
        "Minnesota Vikings": "MIN",
        "New England Patriots": "NE",
        "New Orleans Saints": "NO",
        "New York Giants": "NYG",
        "New York Jets": "NYJ",
        "Philadelphia Eagles": "PHI",
        "Pittsburgh Steelers": "PIT",
        "San Francisco 49ers": "SF",
        "Seattle Seahawks": "SEA",
        "Tampa Bay Buccaneers": "TB",
        "Tennessee Titans": "TEN",
        "Washington Commanders": "WAS",
        "Washington Football Team": "WAS",  # Old name (2020-2021)
        "Washington Redskins": "WAS",  # Very old name
        "Oakland Raiders": "LV",  # Pre-2020
        "San Diego Chargers": "LAC",  # Pre-2017
        "St. Louis Rams": "LAR",  # Pre-2016
    }
    
    # Try exact match first
    if team_name in team_mapping:
        return team_mapping[team_name]
    
    # Try case-insensitive match
    team_name_lower = team_name.lower()
    for api_name, abbrev in team_mapping.items():
        if api_name.lower() == team_name_lower:
            return abbrev
    
    logger.warning(f"Could not map The Odds API team name: {team_name}")
    return None


def fetch_odds_api_historical(
    seasons: List[int],
    api_key: Optional[str] = None,
    regions: str = "us",
    markets: str = "spreads,totals",
) -> pd.DataFrame:
    """
    Fetch historical NFL odds from The Odds API.
    
    The Odds API free tier: 500 requests/month
    Historical endpoint: /v4/sports/{sport}/odds-history/
    
    Args:
        seasons: List of season years to fetch
        api_key: The Odds API key (if None, reads from config)
        regions: Comma-separated regions (default: "us")
        markets: Comma-separated markets (default: "spreads,totals")
    
    Returns:
        DataFrame with odds data, or empty DataFrame if fetch fails
    """
    if not REQUESTS_AVAILABLE:
        logger.warning("requests library not available. Skipping The Odds API fetch.")
        return pd.DataFrame()
    
    # Load API key from config if not provided
    if api_key is None:
        config = load_config()
        odds_config = config.get("nfl", {}).get("odds", {})
        api_config = odds_config.get("the_odds_api", {})
        api_key = api_config.get("api_key")
        
        if not api_key:
            logger.warning("The Odds API key not found in config. Skipping API fetch.")
            return pd.DataFrame()
    
    base_url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds-history"
    
    all_odds = []
    request_count = 0
    max_requests = 450  # Leave some buffer under 500 limit
    
    # NFL season typically runs from early September to early February
    # We'll fetch odds for key dates (game days) to minimize API calls
    for season in seasons:
        logger.info(f"Fetching odds from The Odds API for season {season}...")
        
        # Calculate date range for season
        # Regular season starts early September, ends late December/early January
        season_start = datetime(season, 9, 1)  # September 1
        season_end = datetime(season + 1, 2, 15)  # February 15 (covers Super Bowl)
        
        # Fetch odds for typical game days (Thursdays, Sundays, Mondays)
        # This reduces API calls while covering most games
        current_date = season_start
        dates_fetched = set()
        
        while current_date <= season_end and request_count < max_requests:
            # NFL games are typically on Thu, Sun, Mon
            # Fetch on these days plus a few days before/after to catch all games
            day_of_week = current_date.weekday()  # 0=Monday, 3=Thursday, 6=Sunday
            
            if day_of_week in [0, 3, 6] or (day_of_week == 5):  # Mon, Thu, Sun, Sat
                date_str = current_date.strftime("%Y-%m-%d")
                
                # Skip if already fetched
                if date_str in dates_fetched:
                    current_date += timedelta(days=1)
                    continue
                
                # Build API URL
                url = f"{base_url}"
                params = {
                    "apiKey": api_key,
                    "regions": regions,
                    "markets": markets,
                    "date": date_str,
                }
                
                try:
                    response = requests.get(url, params=params, timeout=10)
                    request_count += 1
                    
                    # Handle rate limiting
                    if response.status_code == 429:
                        remaining = response.headers.get("x-requests-remaining", "0")
                        logger.warning(
                            f"⚠️ Rate limit exceeded for The Odds API. "
                            f"Remaining requests: {remaining}. "
                            f"Stopping API fetch and falling back to CSV/nflverse."
                        )
                        break  # Stop fetching, will fall back to CSV/nflverse
                    
                    response.raise_for_status()
                    data = response.json()
                    
                    # Check remaining requests
                    remaining = response.headers.get("x-requests-remaining")
                    if remaining:
                        remaining_int = int(remaining) if remaining.isdigit() else 0
                        if remaining_int < 50:
                            logger.warning(f"⚠️ Low API requests remaining: {remaining}")
                    
                    # Process response data
                    if isinstance(data, list) and len(data) > 0:
                        for game in data:
                            odds_entry = _parse_odds_api_game(game, season)
                            if odds_entry:
                                all_odds.append(odds_entry)
                        dates_fetched.add(date_str)
                        logger.debug(f"Fetched {len(data)} games for {date_str}")
                    elif isinstance(data, dict) and "data" in data:
                        # Some API versions return wrapped data
                        for game in data.get("data", []):
                            odds_entry = _parse_odds_api_game(game, season)
                            if odds_entry:
                                all_odds.append(odds_entry)
                        dates_fetched.add(date_str)
                    
                    # Small delay to respect rate limits
                    time.sleep(0.3)  # 300ms delay between requests
                    
                except requests.exceptions.RequestException as e:
                    logger.debug(f"Error fetching odds from The Odds API for {date_str}: {e}")
                    # Continue to next date
                    request_count += 1  # Count failed requests too
                
                # Check if we're approaching rate limit
                if request_count >= max_requests:
                    logger.warning(
                        f"⚠️ Approaching API rate limit ({request_count} requests). "
                        f"Stopping fetch and will use available data."
                    )
                    break
            
            current_date += timedelta(days=1)
        
        season_games = len([o for o in all_odds if o.get('season') == season])
        logger.info(f"Fetched odds for {season_games} games from season {season} ({request_count} API requests used)")
    
    if not all_odds:
        logger.warning("No odds data retrieved from The Odds API")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_odds)
    
    if len(df) > 0:
        logger.info(f"Successfully fetched {len(df)} games with odds from The Odds API")
    
    return df


def _parse_odds_api_game(game_data: dict, season: int) -> Optional[dict]:
    """
    Parse a single game from The Odds API response.
    
    Args:
        game_data: Game data dictionary from API response
        season: Season year (for fallback if not in API data)
    
    Returns:
        Dictionary with odds data, or None if parsing fails
    """
    try:
        # Extract teams
        home_team_raw = game_data.get("home_team", "")
        away_team_raw = game_data.get("away_team", "")
        
        home_team = map_odds_api_team_to_abbreviation(home_team_raw)
        away_team = map_odds_api_team_to_abbreviation(away_team_raw)
        
        if not home_team or not away_team:
            logger.debug(f"Skipping game: could not map teams ({away_team_raw} @ {home_team_raw})")
            return None
        
        # Extract date and determine season/week
        commence_time = game_data.get("commence_time")
        if commence_time:
            game_date = pd.to_datetime(commence_time)
            # Determine season and week from date
            # This is approximate - we'll need games_df to get exact week
            if game_date.month >= 9:
                game_season = game_date.year
            else:
                game_season = game_date.year - 1
        else:
            game_season = season
            game_date = None
        
        # Extract odds from bookmakers
        bookmakers = game_data.get("bookmakers", [])
        
        spreads = []
        totals = []
        open_spreads = []
        open_totals = []
        
        for bookmaker in bookmakers:
            markets = bookmaker.get("markets", [])
            
            for market in markets:
                key = market.get("key")
                outcomes = market.get("outcomes", [])
                
                if key == "spreads":
                    # Extract spread (from home team perspective)
                    for outcome in outcomes:
                        if outcome.get("name") == home_team_raw:
                            point = outcome.get("point")
                            if point is not None:
                                spreads.append(float(point))
                                # Check if this is opening line (first bookmaker or marked as opening)
                                if len(spreads) == 1:
                                    open_spreads.append(float(point))
                
                elif key == "totals":
                    # Extract total (over/under)
                    for outcome in outcomes:
                        point = outcome.get("point")
                        if point is not None:
                            totals.append(float(point))
                            # Check if this is opening line
                            if len(totals) == 1:
                                open_totals.append(float(point))
                            break  # Both over/under have same point value
        
        # Use average of all bookmakers for closing line
        # Or use most recent/last bookmaker if available
        close_spread = None
        close_total = None
        open_spread = None
        open_total = None
        
        if spreads:
            # Use average of all spreads (or last one as closing)
            close_spread = sum(spreads) / len(spreads) if len(spreads) > 1 else spreads[-1]
            if open_spreads:
                open_spread = open_spreads[0]
        
        if totals:
            close_total = sum(totals) / len(totals) if len(totals) > 1 else totals[-1]
            if open_totals:
                open_total = open_totals[0]
        
        if close_spread is None or close_total is None:
            logger.debug(f"Skipping game: missing required odds ({away_team} @ {home_team})")
            return None
        
        return {
            "season": game_season,
            "away_team": away_team,
            "home_team": home_team,
            "close_spread": close_spread,
            "close_total": close_total,
            "open_spread": open_spread if open_spread else None,
            "open_total": open_total if open_total else None,
            "game_date": game_date,
        }
    
    except Exception as e:
        logger.error(f"Error parsing odds API game data: {e}")
        return None


def load_odds_from_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load odds from CSV file.
    
    Expected CSV columns:
    - season, week, away_team, home_team, close_spread, close_total
    - Optional: open_spread, open_total
    
    Returns:
        DataFrame with odds data
    """
    logger.info(f"Loading odds from CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Normalize team abbreviations
    if "away_team" in df.columns:
        df["away_team"] = df["away_team"].apply(normalize_team_abbreviation)
    if "home_team" in df.columns:
        df["home_team"] = df["home_team"].apply(normalize_team_abbreviation)
    
    return df


def fetch_nflverse_odds(seasons: List[int]) -> pd.DataFrame:
    """
    Fetch NFL odds from nflverse schedule data.
    
    nflverse schedule data includes spread and total columns if available.
    This function extracts those fields.
    
    Args:
        seasons: List of season years
    
    Returns:
        DataFrame with odds data
    """
    try:
        import nfl_data_py as nfl
    except ImportError:
        logger.warning("nfl_data_py not available for odds fetching")
        return pd.DataFrame()
    
    logger.info(f"Fetching NFL odds from nflverse for seasons {seasons}")
    all_schedules = []
    
    for season in seasons:
        try:
            schedule = nfl.import_schedules([season])
            all_schedules.append(schedule)
        except Exception as e:
            logger.error(f"Error fetching season {season}: {e}")
            continue
    
    if not all_schedules:
        return pd.DataFrame()
    
    df = pd.concat(all_schedules, ignore_index=True)
    
    # Check if spread and total columns exist
    spread_cols = [col for col in df.columns if "spread" in col.lower()]
    total_cols = [col for col in df.columns if "total" in col.lower() or "over_under" in col.lower()]
    
    if not spread_cols or not total_cols:
        logger.warning("nflverse schedule data does not include spread/total columns")
        return pd.DataFrame()
    
    # Use the first matching column for each
    spread_col = spread_cols[0]
    total_col = total_cols[0]
    
    # Extract odds data - ensure we have team columns
    odds_df = pd.DataFrame({
        "season": df["season"],
        "week": df["week"],
        "close_spread": pd.to_numeric(df[spread_col], errors="coerce"),
        "close_total": pd.to_numeric(df[total_col], errors="coerce"),
    })
    
    # Add team columns if available
    if "away_team" in df.columns:
        odds_df["away_team"] = df["away_team"]
    elif "away" in df.columns:
        odds_df["away_team"] = df["away"]
    if "home_team" in df.columns:
        odds_df["home_team"] = df["home_team"]
    elif "home" in df.columns:
        odds_df["home_team"] = df["home"]
    
    # Check for open/close columns
    open_spread_cols = [col for col in df.columns if "open" in col.lower() and "spread" in col.lower()]
    open_total_cols = [col for col in df.columns if "open" in col.lower() and ("total" in col.lower() or "over_under" in col.lower())]
    
    if open_spread_cols:
        odds_df["open_spread"] = pd.to_numeric(df[open_spread_cols[0]], errors="coerce")
    if open_total_cols:
        odds_df["open_total"] = pd.to_numeric(df[open_total_cols[0]], errors="coerce")
    
    # Remove rows with missing required data
    odds_df = odds_df.dropna(subset=["close_spread", "close_total"])
    
    logger.info(f"Extracted {len(odds_df)} games with odds data from nflverse")
    return odds_df


def normalize_to_market_schema(
    df: pd.DataFrame, games_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Normalize odds data to MarketSnapshot schema.
    
    If games_df is provided, uses it to form game_id and validate.
    Otherwise, expects df to have season, week, away_team, home_team.
    
    Args:
        df: DataFrame with odds data
        games_df: Optional DataFrame with games (for game_id matching)
    
    Returns:
        DataFrame with normalized MarketSnapshot schema
    """
    logger.info("Normalizing odds data to MarketSnapshot schema")
    
    # If games_df provided, merge to get game_id
    if games_df is not None:
        # Merge on season, week, away_team, home_team
        merge_cols = ["season", "week", "away_team", "home_team"]
        # Ensure we have all merge columns in both dataframes
        available_merge_cols = [col for col in merge_cols if col in df.columns and col in games_df.columns]
        
        if "game_id" not in df.columns:
            if len(available_merge_cols) >= 2:  # At least season and week
                # Merge to get game_id, preserving season and week
                df = df.merge(
                    games_df[["game_id"] + merge_cols],
                    on=merge_cols,
                    how="left",
                )
            else:
                logger.warning("Cannot merge on team names, missing required columns")
        
        # Check for unmatched games
        unmatched = df[df["game_id"].isna()] if "game_id" in df.columns else pd.DataFrame()
        if len(unmatched) > 0:
            logger.warning(f"{len(unmatched)} odds entries could not be matched to games")
    else:
        # Form game_id from available columns
        if all(col in df.columns for col in ["season", "week", "away_team", "home_team"]):
            df["game_id"] = df.apply(
                lambda row: form_game_id(
                    row["season"], row["week"], row["away_team"], row["home_team"]
                ),
                axis=1,
            )
        else:
            raise ValueError("Cannot form game_id: missing required columns")
    
    # Normalize team abbreviations
    if "away_team" in df.columns:
        df["away_team"] = df["away_team"].apply(normalize_team_abbreviation)
    if "home_team" in df.columns:
        df["home_team"] = df["home_team"].apply(normalize_team_abbreviation)
    
    # Ensure required columns exist
    required_cols = ["game_id", "close_spread", "close_total"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Build MarketSnapshot schema
    # Ensure season and week are included (required for join and validation)
    if "game_id" not in df.columns:
        raise ValueError("game_id is required in df or games_df must be provided")
    
    market_df = pd.DataFrame({
        "game_id": df["game_id"],
        "close_spread": pd.to_numeric(df["close_spread"], errors="coerce"),
        "close_total": pd.to_numeric(df["close_total"], errors="coerce"),
    })
    
    # Add season and week - they should be in df after merge or from original data
    if "season" in df.columns:
        market_df["season"] = df["season"]
    elif games_df is not None and "season" in games_df.columns:
        # If we merged with games_df, get season from there
        season_lookup = games_df.set_index("game_id")["season"].to_dict()
        market_df["season"] = market_df["game_id"].map(season_lookup)
    else:
        raise ValueError("season column is required")
    
    if "week" in df.columns:
        market_df["week"] = df["week"]
    elif games_df is not None and "week" in games_df.columns:
        # If we merged with games_df, get week from there
        week_lookup = games_df.set_index("game_id")["week"].to_dict()
        market_df["week"] = market_df["game_id"].map(week_lookup)
    else:
        raise ValueError("week column is required")
    
    # Add optional columns if available
    if "open_spread" in df.columns:
        market_df["open_spread"] = pd.to_numeric(df["open_spread"], errors="coerce")
    if "open_total" in df.columns:
        market_df["open_total"] = pd.to_numeric(df["open_total"], errors="coerce")
    
    # Remove rows with missing required data
    initial_len = len(market_df)
    market_df = market_df.dropna(subset=["game_id", "close_spread", "close_total"])
    if len(market_df) < initial_len:
        logger.warning(f"Dropped {initial_len - len(market_df)} rows with missing required data")
    
    # Sort by season, week
    market_df = market_df.sort_values(["season", "week"]).reset_index(drop=True)
    
    logger.info(f"Normalized {len(market_df)} market entries")
    return market_df


def ingest_nfl_odds(
    seasons: Optional[List[int]] = None,
    csv_path: Optional[Path] = None,
    games_df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Main ingestion function for NFL odds.
    
    Tries data sources in priority order:
    1. The Odds API (free tier: 500 requests/month) - PRIMARY SOURCE
    2. CSV file (manual fallback) - FALLBACK 1
    3. nflverse schedule data (last resort) - FALLBACK 2
    
    All sources produce consistent schema output:
    - game_id: nfl_{season}_{week}_{away}_{home}
    - season: Season year
    - week: Week number
    - away_team: Away team abbreviation
    - home_team: Home team abbreviation
    - close_spread: Closing spread (from home team perspective)
    - close_total: Closing total (over/under)
    - open_spread: Opening spread (optional)
    - open_total: Opening total (optional)
    
    Args:
        seasons: List of seasons to fetch. If None, uses config.
        csv_path: Optional path to CSV file with odds data (used as fallback).
        games_df: Optional DataFrame with games (for validation/matching).
        output_path: Path to save parquet file. If None, uses default.
    
    Returns:
        DataFrame with normalized MarketSnapshot schema (game_id, season, week, close_spread, close_total, ...)
    """
    logger.info("=" * 60)
    logger.info("NFL Odds Ingestion - Starting")
    logger.info("=" * 60)
    
    # Load configuration
    config = load_config()
    
    if seasons is None:
        seasons = config["nfl"]["schedule"]["seasons"]
    logger.info(f"Target seasons: {seasons}")
    
    if output_path is None:
        output_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "nfl"
            / "staged"
            / "markets.parquet"
        )
    
    odds_df = pd.DataFrame()
    source_used = None
    fallback_reasons = []
    
    # ========================================================================
    # PRIORITY 1: The Odds API (PRIMARY SOURCE)
    # ========================================================================
    logger.info("\n" + "-" * 60)
    logger.info("PRIORITY 1: Attempting The Odds API")
    logger.info("-" * 60)
    
    odds_config = config.get("nfl", {}).get("odds", {})
    api_config = odds_config.get("the_odds_api", {})
    api_enabled = api_config.get("enabled", True)
    api_key = api_config.get("api_key", "")
    
    if not api_enabled:
        fallback_reasons.append("The Odds API is disabled in config (enabled: false)")
        logger.info("⚠️ The Odds API is disabled in configuration")
    elif not api_key:
        fallback_reasons.append("The Odds API key not found in config")
        logger.info("⚠️ The Odds API key not configured")
    elif not REQUESTS_AVAILABLE:
        fallback_reasons.append("requests library not available")
        logger.warning("⚠️ requests library not available. Install with: pip install requests")
    else:
        logger.info(f"✅ The Odds API configured (key: {api_key[:8]}...{api_key[-4:]})")
        logger.info(f"   Regions: {api_config.get('regions', 'us')}")
        logger.info(f"   Markets: {api_config.get('markets', 'spreads,totals')}")
        
        try:
            logger.info("Making API request to fetch historical odds...")
            odds_df = fetch_odds_api_historical(
                seasons=seasons,
                api_key=api_key,
                regions=api_config.get("regions", "us"),
                markets=api_config.get("markets", "spreads,totals"),
            )
            
            if len(odds_df) > 0:
                source_used = "the_odds_api"
                logger.info(f"✅ SUCCESS: Fetched {len(odds_df)} games from The Odds API")
                logger.info(f"   Schema columns: {list(odds_df.columns)}")
            else:
                fallback_reason = "The Odds API returned empty result (no games found)"
                fallback_reasons.append(fallback_reason)
                logger.warning(f"⚠️ {fallback_reason}")
                logger.info("   → Falling back to CSV file...")
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                fallback_reason = f"The Odds API authentication failed (401): Invalid API key"
                fallback_reasons.append(fallback_reason)
                logger.error(f"❌ {fallback_reason}")
            elif e.response.status_code == 429:
                fallback_reason = f"The Odds API rate limit exceeded (429): Too many requests"
                fallback_reasons.append(fallback_reason)
                logger.warning(f"⚠️ {fallback_reason}")
            else:
                fallback_reason = f"The Odds API HTTP error ({e.response.status_code}): {str(e)}"
                fallback_reasons.append(fallback_reason)
                logger.error(f"❌ {fallback_reason}")
            logger.info("   → Falling back to CSV file...")
            
        except requests.exceptions.Timeout as e:
            fallback_reason = f"The Odds API request timeout: {str(e)}"
            fallback_reasons.append(fallback_reason)
            logger.warning(f"⚠️ {fallback_reason}")
            logger.info("   → Falling back to CSV file...")
            
        except requests.exceptions.ConnectionError as e:
            fallback_reason = f"The Odds API connection error: {str(e)}"
            fallback_reasons.append(fallback_reason)
            logger.warning(f"⚠️ {fallback_reason}")
            logger.info("   → Falling back to CSV file...")
            
        except Exception as e:
            fallback_reason = f"The Odds API error: {type(e).__name__}: {str(e)}"
            fallback_reasons.append(fallback_reason)
            logger.error(f"❌ Unexpected error from The Odds API: {e}")
            logger.info("   → Falling back to CSV file...")
    
    # ========================================================================
    # PRIORITY 2: CSV File (FALLBACK 1)
    # ========================================================================
    if len(odds_df) == 0:
        logger.info("\n" + "-" * 60)
        logger.info("PRIORITY 2: Attempting CSV File Fallback")
        logger.info("-" * 60)
        
        if csv_path is None:
            fallback_reason = "CSV path not provided"
            fallback_reasons.append(fallback_reason)
            logger.info(f"⚠️ {fallback_reason}")
            logger.info("   → Falling back to nflverse...")
        elif not csv_path.exists():
            fallback_reason = f"CSV file not found: {csv_path}"
            fallback_reasons.append(fallback_reason)
            logger.warning(f"⚠️ {fallback_reason}")
            logger.info("   → Falling back to nflverse...")
        else:
            logger.info(f"Loading odds from CSV file: {csv_path}")
            try:
                odds_df = load_odds_from_csv(csv_path)
                
                if len(odds_df) > 0:
                    source_used = "csv"
                    logger.info(f"✅ SUCCESS: Loaded {len(odds_df)} games from CSV file")
                    logger.info(f"   Schema columns: {list(odds_df.columns)}")
                else:
                    fallback_reason = "CSV file is empty or contains no valid data"
                    fallback_reasons.append(fallback_reason)
                    logger.warning(f"⚠️ {fallback_reason}")
                    logger.info("   → Falling back to nflverse...")
                    
            except pd.errors.EmptyDataError as e:
                fallback_reason = f"CSV file is empty: {str(e)}"
                fallback_reasons.append(fallback_reason)
                logger.warning(f"⚠️ {fallback_reason}")
                logger.info("   → Falling back to nflverse...")
                
            except pd.errors.ParserError as e:
                fallback_reason = f"CSV parsing error: {str(e)}"
                fallback_reasons.append(fallback_reason)
                logger.error(f"❌ {fallback_reason}")
                logger.info("   → Falling back to nflverse...")
                
            except FileNotFoundError as e:
                fallback_reason = f"CSV file not found: {str(e)}"
                fallback_reasons.append(fallback_reason)
                logger.error(f"❌ {fallback_reason}")
                logger.info("   → Falling back to nflverse...")
                
            except Exception as e:
                fallback_reason = f"CSV loading error: {type(e).__name__}: {str(e)}"
                fallback_reasons.append(fallback_reason)
                logger.error(f"❌ Unexpected error loading CSV: {e}")
                logger.info("   → Falling back to nflverse...")
    
    # ========================================================================
    # PRIORITY 3: nflverse (FALLBACK 2 - LAST RESORT)
    # ========================================================================
    if len(odds_df) == 0:
        logger.info("\n" + "-" * 60)
        logger.info("PRIORITY 3: Attempting nflverse Fallback (Last Resort)")
        logger.info("-" * 60)
        
        logger.info("Extracting odds from nflverse schedule data...")
        try:
            odds_df = fetch_nflverse_odds(seasons)
            
            if len(odds_df) > 0:
                source_used = "nflverse"
                logger.info(f"✅ SUCCESS: Extracted {len(odds_df)} games from nflverse")
                logger.info(f"   Schema columns: {list(odds_df.columns)}")
                logger.warning("⚠️ Note: nflverse may not have odds for all seasons/games")
            else:
                fallback_reason = "nflverse returned no odds data (spread/total columns may not exist)"
                fallback_reasons.append(fallback_reason)
                logger.warning(f"⚠️ {fallback_reason}")
                
        except ImportError as e:
            fallback_reason = f"nflverse library not available: {str(e)}"
            fallback_reasons.append(fallback_reason)
            logger.error(f"❌ {fallback_reason}")
            logger.error("   Install with: pip install nfl-data-py")
            
        except Exception as e:
            fallback_reason = f"nflverse error: {type(e).__name__}: {str(e)}"
            fallback_reasons.append(fallback_reason)
            logger.error(f"❌ Unexpected error from nflverse: {e}")
    
    # ========================================================================
    # FINAL VALIDATION AND ERROR HANDLING
    # ========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Ingestion Summary")
    logger.info("=" * 60)
    
    if len(odds_df) == 0:
        error_msg = (
            "❌ FAILED: No odds data available from any source.\n"
            f"   Attempted sources:\n"
            f"   1. The Odds API\n"
            f"   2. CSV file{' (not provided)' if csv_path is None else f' ({csv_path})'}\n"
            f"   3. nflverse\n"
            f"\n"
            f"   Fallback reasons:\n"
        )
        for i, reason in enumerate(fallback_reasons, 1):
            error_msg += f"   {i}. {reason}\n"
        error_msg += (
            f"\n"
            f"   Recommendations:\n"
            f"   - Verify The Odds API key in config/data/nfl.yaml\n"
            f"   - Provide a CSV file with historical odds data\n"
            f"   - Check nflverse data availability\n"
            f"   - See docs/data_sources.md for more information"
        )
        logger.error(error_msg)
        raise ValueError("No odds data available from any source")
    
    # Log successful source
    logger.info(f"✅ Data source used: {source_used.upper()}")
    logger.info(f"   Games retrieved: {len(odds_df)}")
    logger.info(f"   Schema: {list(odds_df.columns)}")
    
    if fallback_reasons:
        logger.info(f"\n⚠️ Fallback occurred - reasons:")
        for i, reason in enumerate(fallback_reasons, 1):
            logger.info(f"   {i}. {reason}")
    
    # Ensure consistent schema - normalize to MarketSnapshot schema
    logger.info("\nNormalizing data to MarketSnapshot schema...")
    market_df = normalize_to_market_schema(odds_df, games_df)
    
    # Validate schema consistency
    required_cols = ["game_id", "season", "week", "close_spread", "close_total"]
    missing_cols = [col for col in required_cols if col not in market_df.columns]
    if missing_cols:
        logger.error(f"❌ Schema validation failed. Missing columns: {missing_cols}")
        raise ValueError(f"Invalid schema: missing required columns {missing_cols}")
    
    logger.info(f"✅ Schema validated. Columns: {list(market_df.columns)}")
    logger.info(f"   Required columns present: {required_cols}")
    
    # Save raw data
    raw_output_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "nfl"
        / "raw"
        / "odds.parquet"
    )
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    odds_df.to_parquet(raw_output_path, index=False)
    logger.info(f"\n💾 Saved raw odds data to: {raw_output_path}")
    logger.info(f"   Rows: {len(odds_df)}")
    
    # Save staged data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    market_df.to_parquet(output_path, index=False)
    logger.info(f"💾 Saved normalized odds data to: {output_path}")
    logger.info(f"   Rows: {len(market_df)}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ NFL Odds Ingestion Complete")
    logger.info("=" * 60)
    logger.info(f"   Source: {source_used}")
    logger.info(f"   Games: {len(market_df)}")
    logger.info(f"   Output: {output_path}")
    
    return market_df


def test_fetch_odds_api():
    """
    Test function to verify The Odds API integration.
    
    Loads API key from config, makes a sample request, and displays results.
    Handles errors gracefully (rate limiting, invalid key, bad response).
    """
    print("=" * 60)
    print("Testing The Odds API Integration")
    print("=" * 60)
    
    # Load config
    try:
        config = load_config()
        odds_config = config.get("nfl", {}).get("odds", {})
        api_config = odds_config.get("the_odds_api", {})
        api_key = api_config.get("api_key", "")
        enabled = api_config.get("enabled", True)
        regions = api_config.get("regions", "us")
        markets = api_config.get("markets", "spreads,totals")
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        print(f"❌ Failed to load config: {e}")
        return
    
    # Check if API is enabled
    if not enabled:
        print("⚠️ The Odds API is disabled in config (enabled: false)")
        return
    
    # Check if API key is configured
    if not api_key:
        print("⚠️ API key not found in config. Using mock endpoint for testing...")
        print("   To test with real API, set api_key in config/data/nfl.yaml")
        # Use mock endpoint or return early
        print("   Skipping API test (no API key configured)")
        return
    
    # Check if requests library is available
    if not REQUESTS_AVAILABLE:
        print("❌ requests library not available. Install with: pip install requests")
        return
    
    print(f"✅ API key found: {api_key[:8]}...{api_key[-4:]}")
    print(f"   Regions: {regions}")
    print(f"   Markets: {markets}")
    print()
    
    # Make sample request to The Odds API
    # Use current odds endpoint (not historical) for testing - requires fewer params
    base_url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
    }
    
    print("Making test request to The Odds API...")
    print(f"URL: {base_url}")
    print()
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        
        # Check response status
        if response.status_code == 401:
            logger.error("Invalid API key")
            print("❌ Error: Invalid API key (401 Unauthorized)")
            print("   Please check your API key in config/data/nfl.yaml")
            return
        
        elif response.status_code == 429:
            remaining = response.headers.get("x-requests-remaining", "unknown")
            reset_time = response.headers.get("x-request-reset-time", "unknown")
            logger.warning(f"Rate limit exceeded. Remaining: {remaining}, Reset: {reset_time}")
            print(f"⚠️ Rate limit exceeded (429 Too Many Requests)")
            print(f"   Remaining requests: {remaining}")
            print(f"   Reset time: {reset_time}")
            return
        
        elif response.status_code != 200:
            logger.error(f"API returned status {response.status_code}: {response.text}")
            print(f"❌ Error: API returned status {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return
        
        # Parse JSON response
        try:
            data = response.json()
        except ValueError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            print(f"❌ Error: Failed to parse JSON response: {e}")
            print(f"   Response text: {response.text[:200]}")
            return
        
        # Check remaining requests
        remaining = response.headers.get("x-requests-remaining")
        if remaining:
            remaining_int = int(remaining) if remaining.isdigit() else 0
            print(f"✅ API request successful!")
            print(f"   Remaining requests this month: {remaining}")
            if remaining_int < 50:
                print(f"   ⚠️ Warning: Low remaining requests ({remaining})")
            print()
        
        # Process response data
        if not isinstance(data, list):
            logger.error(f"Unexpected response format: {type(data)}")
            print(f"❌ Error: Unexpected response format (expected list, got {type(data)})")
            print(f"   Response: {str(data)[:200]}")
            return
        
        if len(data) == 0:
            print("⚠️ No games found in API response (may be off-season)")
            print("   This is normal if there are no upcoming NFL games")
            return
        
        print(f"✅ Found {len(data)} games in API response")
        print()
        
        # Parse first 3 games and display as DataFrame
        parsed_games = []
        for i, game in enumerate(data[:3]):
            try:
                parsed = _parse_odds_api_game(game, datetime.now().year)
                if parsed:
                    parsed_games.append(parsed)
            except Exception as e:
                logger.debug(f"Error parsing game {i}: {e}")
                continue
        
        if not parsed_games:
            print("⚠️ Could not parse any games from API response")
            print("   Raw response structure:")
            if len(data) > 0:
                import json
                print(json.dumps(data[0], indent=2)[:500])
            return
        
        # Create DataFrame
        df = pd.DataFrame(parsed_games)
        
        # Display results
        print("=" * 60)
        print("First 3 Games (Parsed)")
        print("=" * 60)
        print(df.to_string(index=False))
        print()
        
        # Show summary
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Total games in response: {len(data)}")
        print(f"Successfully parsed: {len(parsed_games)}")
        print(f"Columns: {', '.join(df.columns)}")
        print()
        
        # Show sample raw data structure
        if len(data) > 0:
            print("=" * 60)
            print("Sample Raw API Response (first game)")
            print("=" * 60)
            import json
            sample_game = data[0]
            # Remove bookmakers for brevity (they're large)
            if "bookmakers" in sample_game:
                sample_game_display = {k: v for k, v in sample_game.items() if k != "bookmakers"}
                sample_game_display["bookmakers"] = f"[{len(sample_game['bookmakers'])} bookmakers]"
            else:
                sample_game_display = sample_game
            print(json.dumps(sample_game_display, indent=2, default=str))
            print()
        
        print("✅ Test completed successfully!")
        
    except requests.exceptions.Timeout:
        logger.error("Request timeout")
        print("❌ Error: Request timeout (API may be slow or unavailable)")
        
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error: {e}")
        print(f"❌ Error: Connection failed - {e}")
        print("   Check your internet connection")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        print(f"❌ Error: Request failed - {e}")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"❌ Unexpected error: {e}")
        print("   Check logs for details")


if __name__ == "__main__":
    import sys
    
    # If test flag is provided, run test function
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_fetch_odds_api()
    else:
        # Original usage - requires CSV file or games_df
        if len(sys.argv) > 1:
            csv_path = Path(sys.argv[1])
            df = ingest_nfl_odds(csv_path=csv_path)
        else:
            # Default: run test function
            print("Running The Odds API test...")
            print("(Use --test flag explicitly, or provide CSV path as argument)")
            print()
            test_fetch_odds_api()

