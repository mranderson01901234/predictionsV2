"""
NFL Schedule and Results Ingestion Module

Fetches NFL schedules and final scores from nflverse/nflfastR data sources
and normalizes to the core Game schema.
"""

import pandas as pd
import yaml
from pathlib import Path
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load NFL data configuration."""
    config_path = Path(__file__).parent.parent.parent / "config" / "data" / "nfl.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def normalize_team_abbreviation(team: str, season: int = None) -> str:
    """
    Normalize team abbreviations to nflverse standard.
    
    nflverse uses 3-letter abbreviations. This function ensures consistency.
    
    Args:
        team: Team abbreviation to normalize
        season: Optional season year for context (helps with LA teams)
    """
    # nflverse team abbreviations mapping (standard NFL abbreviations)
    team_map = {
        "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BUF": "BUF",
        "CAR": "CAR", "CHI": "CHI", "CIN": "CIN", "CLE": "CLE",
        "DAL": "DAL", "DEN": "DEN", "DET": "DET", "GB": "GB",
        "HOU": "HOU", "IND": "IND", "JAX": "JAX", "KC": "KC",
        "LV": "LV", "LAR": "LAR", "LAC": "LAC", "MIA": "MIA",
        "MIN": "MIN", "NE": "NE", "NO": "NO", "NYG": "NYG",
        "NYJ": "NYJ", "PHI": "PHI", "PIT": "PIT", "SF": "SF",
        "SEA": "SEA", "TB": "TB", "TEN": "TEN", "WAS": "WAS",
        # Handle variations
        "OAK": "LV",  # Raiders moved from Oakland to Las Vegas (2020)
        "SD": "LAC",  # Chargers moved from San Diego to Los Angeles (2017)
        "STL": "LAR",  # Rams moved from St. Louis to Los Angeles (2016)
        # Handle ambiguous "LA" abbreviation (used 2016-2019 before teams were distinguished)
        # Note: This is a best-effort mapping. In practice, nflverse should use LAR/LAC
        "LA": "LAR",  # Default to Rams (most common), but this should be rare
    }
    
    # If already normalized, return as-is
    if team in team_map:
        result = team_map[team]
        # Special handling for LA: if season is 2016-2019, we can't distinguish
        # But nflverse should use LAR/LAC, so this is a fallback
        if team == "LA" and season:
            # This shouldn't happen with modern nflverse data, but handle it
            logger.warning(f"Ambiguous 'LA' team abbreviation for season {season}, defaulting to LAR")
        return result
    
    # Try uppercase
    team_upper = team.upper()
    if team_upper in team_map:
        return team_map[team_upper]
    
    # If not found, log warning and return original
    logger.warning(f"Unknown team abbreviation: {team}, using as-is")
    return team.upper()[:3] if len(team) >= 3 else team.upper()


def form_game_id(season: int, week: int, away_team: str, home_team: str) -> str:
    """
    Form game_id in format: nfl_{season}_{week}_{away}_{home}
    
    Args:
        season: NFL season year
        week: Week number (1-18 for regular season, 19+ for playoffs)
        away_team: Away team abbreviation (normalized)
        home_team: Home team abbreviation (normalized)
    
    Returns:
        Formatted game_id string
    """
    away_norm = normalize_team_abbreviation(away_team)
    home_norm = normalize_team_abbreviation(home_team)
    return f"nfl_{season}_{week:02d}_{away_norm}_{home_norm}"


def fetch_nflverse_schedules(seasons: List[int]) -> pd.DataFrame:
    """
    Fetch NFL schedules from nflverse.
    
    Args:
        seasons: List of season years to fetch
    
    Returns:
        DataFrame with schedule data
    """
    try:
        import nfl_data_py as nfl
    except ImportError:
        raise ImportError(
            "nfl_data_py (nflverse) is required. Install with: pip install nfl-data-py"
        )
    
    logger.info(f"Fetching NFL schedules for seasons {seasons}")
    all_schedules = []
    
    for season in seasons:
        try:
            schedule = nfl.import_schedules([season])
            all_schedules.append(schedule)
            logger.info(f"Fetched {len(schedule)} games for season {season}")
        except Exception as e:
            logger.error(f"Error fetching season {season}: {e}")
            raise
    
    df = pd.concat(all_schedules, ignore_index=True)
    logger.info(f"Total games fetched: {len(df)}")
    return df


def normalize_to_game_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize nflverse schedule data to Game schema.
    
    Handles various column name variations from nflverse.
    
    Returns:
        DataFrame with normalized Game schema columns
    """
    logger.info("Normalizing schedule data to Game schema")
    
    # nflverse column name mappings (handle variations)
    date_cols = ["gameday", "game_date", "date", "game_day"]
    home_team_cols = ["home_team", "home", "home_abbr"]
    away_team_cols = ["away_team", "away", "away_abbr"]
    home_score_cols = ["home_score", "home_points", "home_pts"]
    away_score_cols = ["away_score", "away_points", "away_pts"]
    
    # Find matching columns
    date_col = None
    for col in date_cols:
        if col in df.columns:
            date_col = col
            break
    
    home_team_col = None
    for col in home_team_cols:
        if col in df.columns:
            home_team_col = col
            break
    
    away_team_col = None
    for col in away_team_cols:
        if col in df.columns:
            away_team_col = col
            break
    
    home_score_col = None
    for col in home_score_cols:
        if col in df.columns:
            home_score_col = col
            break
    
    away_score_col = None
    for col in away_score_cols:
        if col in df.columns:
            away_score_col = col
            break
    
    # Validate required columns exist
    missing = []
    if "season" not in df.columns:
        missing.append("season")
    if "week" not in df.columns:
        missing.append("week")
    if date_col is None:
        missing.append("date (gameday/game_date)")
    if home_team_col is None:
        missing.append("home_team")
    if away_team_col is None:
        missing.append("away_team")
    if home_score_col is None:
        missing.append("home_score")
    if away_score_col is None:
        missing.append("away_score")
    
    if missing:
        available = ", ".join(df.columns[:10])
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {available}..."
        )
    
    # Normalize team abbreviations (pass season for context)
    df["home_team_norm"] = df.apply(
        lambda row: normalize_team_abbreviation(row[home_team_col], row.get("season")),
        axis=1
    )
    df["away_team_norm"] = df.apply(
        lambda row: normalize_team_abbreviation(row[away_team_col], row.get("season")),
        axis=1
    )
    
    # Form our game_id
    df["game_id"] = df.apply(
        lambda row: form_game_id(
            row["season"], row["week"], row["away_team_norm"], row["home_team_norm"]
        ),
        axis=1,
    )
    
    # Build Game schema DataFrame
    game_df = pd.DataFrame({
        "game_id": df["game_id"],
        "season": df["season"],
        "week": df["week"],
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "home_team": df["home_team_norm"],
        "away_team": df["away_team_norm"],
        "home_score": pd.to_numeric(df[home_score_col], errors="coerce").fillna(0).astype(int),
        "away_score": pd.to_numeric(df[away_score_col], errors="coerce").fillna(0).astype(int),
    })
    
    # Remove rows with invalid dates (future games or parsing errors)
    initial_len = len(game_df)
    game_df = game_df[game_df["date"].notna()]
    if len(game_df) < initial_len:
        logger.warning(f"Dropped {initial_len - len(game_df)} rows with invalid dates")
    
    # Sort by season, week, date
    game_df = game_df.sort_values(["season", "week", "date"]).reset_index(drop=True)
    
    logger.info(f"Normalized {len(game_df)} games")
    return game_df


def ingest_nfl_schedules(
    seasons: Optional[List[int]] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Main ingestion function for NFL schedules.
    
    Args:
        seasons: List of seasons to fetch. If None, uses config.
        output_path: Path to save parquet file. If None, uses default.
    
    Returns:
        DataFrame with normalized Game schema
    """
    config = load_config()
    
    if seasons is None:
        seasons = config["nfl"]["schedule"]["seasons"]
    
    if output_path is None:
        output_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "nfl"
            / "staged"
            / "games.parquet"
        )
    
    # Fetch data
    raw_df = fetch_nflverse_schedules(seasons)
    
    # Save raw data
    raw_output_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "nfl"
        / "raw"
        / "schedules.parquet"
    )
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_parquet(raw_output_path, index=False)
    logger.info(f"Saved raw schedule data to {raw_output_path}")
    
    # Normalize
    game_df = normalize_to_game_schema(raw_df)
    
    # Save staged data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    game_df.to_parquet(output_path, index=False)
    logger.info(f"Saved normalized schedule data to {output_path}")
    
    return game_df


if __name__ == "__main__":
    # Run ingestion
    df = ingest_nfl_schedules()
    print(f"\nIngested {len(df)} games")
    print(f"\nSeasons: {df['season'].min()} - {df['season'].max()}")
    print(f"\nSample games:")
    print(df.head(10))

