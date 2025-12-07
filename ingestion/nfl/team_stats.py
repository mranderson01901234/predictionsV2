"""
NFL Team Stats Ingestion Module

Fetches team-level game statistics from nflverse and normalizes to TeamGameStats schema.
"""

import pandas as pd
import yaml
from pathlib import Path
from typing import List, Optional
import logging
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ingestion.nfl.schedule import normalize_team_abbreviation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load NFL data configuration."""
    config_path = Path(__file__).parent.parent.parent / "config" / "data" / "nfl.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def fetch_nflverse_team_stats(seasons: List[int]) -> pd.DataFrame:
    """
    Fetch NFL team-level game statistics from nflverse.
    
    Args:
        seasons: List of season years to fetch
    
    Returns:
        DataFrame with team game stats
    """
    try:
        import nfl_data_py as nfl
    except ImportError:
        raise ImportError(
            "nfl_data_py (nflverse) is required. Install with: pip install nfl-data-py"
        )
    
    logger.info(f"Fetching NFL team stats for seasons {seasons}")
    all_stats = []
    
    for season in seasons:
        try:
            # Fetch team stats from nflverse
            # nflverse provides team stats via import_team_desc() or similar
            # We'll use schedule data and calculate stats, or use available endpoints
            schedule = nfl.import_schedules([season])
            
            # Extract team stats from schedule (points, basic stats)
            # For more detailed stats, we may need to use play-by-play data
            # For Phase 1B, we'll work with what's available in schedule
            
            # Get team stats from schedule - each row has home/away teams and scores
            home_stats = pd.DataFrame({
                "season": schedule["season"],
                "week": schedule["week"],
                "game_id_nflverse": schedule.get("game_id", ""),
                "team": schedule.get("home_team", ""),
                "is_home": True,
                "opponent": schedule.get("away_team", ""),
                "points_for": pd.to_numeric(schedule.get("home_score", 0), errors="coerce").fillna(0),
                "points_against": pd.to_numeric(schedule.get("away_score", 0), errors="coerce").fillna(0),
            })
            
            away_stats = pd.DataFrame({
                "season": schedule["season"],
                "week": schedule["week"],
                "game_id_nflverse": schedule.get("game_id", ""),
                "team": schedule.get("away_team", ""),
                "is_home": False,
                "opponent": schedule.get("home_team", ""),
                "points_for": pd.to_numeric(schedule.get("away_score", 0), errors="coerce").fillna(0),
                "points_against": pd.to_numeric(schedule.get("home_score", 0), errors="coerce").fillna(0),
            })
            
            season_stats = pd.concat([home_stats, away_stats], ignore_index=True)
            all_stats.append(season_stats)
            logger.info(f"Fetched team stats for {len(season_stats)} team-games in season {season}")
        except Exception as e:
            logger.error(f"Error fetching team stats for season {season}: {e}")
            raise
    
    df = pd.concat(all_stats, ignore_index=True)
    logger.info(f"Total team-game stats fetched: {len(df)}")
    return df


def fetch_detailed_team_stats(seasons: List[int]) -> pd.DataFrame:
    """
    Fetch more detailed team stats including turnovers, yards, etc.
    
    Attempts to get detailed stats from nflverse or calculates from play-by-play.
    For Phase 1B, we'll try to get what's available.
    """
    try:
        import nfl_data_py as nfl
    except ImportError:
        return pd.DataFrame()
    
    logger.info("Attempting to fetch detailed team stats...")
    
    # Try to get team stats from nflverse
    # Note: nflverse may have different endpoints for detailed stats
    # For now, we'll work with basic stats and add turnovers/yards if available
    
    all_detailed = []
    
    for season in seasons:
        try:
            # Try importing team stats if available
            # nflverse structure may vary, so we'll be flexible
            schedule = nfl.import_schedules([season])
            
            # Check if detailed stats columns exist
            turnover_cols = [col for col in schedule.columns if "turnover" in col.lower()]
            yard_cols = [col for col in schedule.columns if "yard" in col.lower() and "total" in col.lower()]
            
            detailed = pd.DataFrame({
                "season": schedule["season"],
                "week": schedule["week"],
                "game_id_nflverse": schedule.get("game_id", ""),
            })
            
            # Add turnovers if available
            if turnover_cols:
                logger.info(f"Found turnover columns: {turnover_cols}")
                # Would need to map home/away turnovers appropriately
            
            all_detailed.append(detailed)
        except Exception as e:
            logger.warning(f"Could not fetch detailed stats for season {season}: {e}")
            continue
    
    if all_detailed:
        return pd.concat(all_detailed, ignore_index=True)
    return pd.DataFrame()


def normalize_to_team_stats_schema(
    df: pd.DataFrame, games_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Normalize team stats data to TeamGameStats schema.
    
    Args:
        df: DataFrame with team stats from nflverse
        games_df: Optional DataFrame with games (for game_id matching)
    
    Returns:
        DataFrame with normalized TeamGameStats schema
    """
    logger.info("Normalizing team stats to TeamGameStats schema")
    
    # Normalize team abbreviations
    df["team"] = df["team"].apply(normalize_team_abbreviation)
    if "opponent" in df.columns:
        df["opponent"] = df["opponent"].apply(normalize_team_abbreviation)
    
    # If games_df provided, form game_id to match
    if games_df is not None:
        # Form game_id from schedule data to match games_df format
        from ingestion.nfl.schedule import form_game_id
        
        # Add game_id to df based on season, week, teams
        # Our format: nfl_{season}_{week}_{away}_{home}
        # If is_home=True: team is home, opponent is away
        # If is_home=False: team is away, opponent is home
        df["game_id"] = df.apply(
            lambda row: form_game_id(
                int(row["season"]),
                int(row["week"]),
                row["opponent"] if row["is_home"] else row["team"],  # away team
                row["team"] if row["is_home"] else row["opponent"],  # home team
            ),
            axis=1,
        )
        
        # Verify game_ids exist in games_df
        valid_game_ids = set(games_df["game_id"].unique())
        df_game_ids = set(df["game_id"].unique())
        missing = df_game_ids - valid_game_ids
        
        if len(missing) > 0:
            logger.warning(f"{len(missing)} game_ids from stats not found in games_df")
            # Show sample of missing for debugging
            if len(missing) < 10:
                logger.debug(f"Missing game_ids: {list(missing)}")
            # Filter to only valid game_ids
            df = df[df["game_id"].isin(valid_game_ids)]
    
    # Build TeamGameStats schema
    team_stats_df = pd.DataFrame({
        "game_id": df["game_id"],
        "team": df["team"],
        "is_home": df["is_home"].astype(bool),
        "points_for": pd.to_numeric(df["points_for"], errors="coerce").fillna(0).astype(int),
        "points_against": pd.to_numeric(df["points_against"], errors="coerce").fillna(0).astype(int),
    })
    
    # Add optional fields if available
    if "turnovers" in df.columns:
        team_stats_df["turnovers"] = pd.to_numeric(df["turnovers"], errors="coerce").fillna(0).astype(int)
    else:
        team_stats_df["turnovers"] = 0  # Placeholder - will be filled from play-by-play in Phase 2
    
    if "yards_for" in df.columns or "total_yards" in df.columns:
        yard_col = "yards_for" if "yards_for" in df.columns else "total_yards"
        team_stats_df["yards_for"] = pd.to_numeric(df[yard_col], errors="coerce").fillna(0).astype(int)
    else:
        team_stats_df["yards_for"] = 0  # Placeholder
    
    if "yards_against" in df.columns:
        team_stats_df["yards_against"] = pd.to_numeric(df["yards_against"], errors="coerce").fillna(0).astype(int)
    else:
        team_stats_df["yards_against"] = 0  # Placeholder
    
    # Remove rows with missing game_id
    initial_len = len(team_stats_df)
    team_stats_df = team_stats_df[team_stats_df["game_id"].notna()]
    if len(team_stats_df) < initial_len:
        logger.warning(f"Dropped {initial_len - len(team_stats_df)} rows with missing game_id")
    
    # Sort by game_id, is_home
    team_stats_df = team_stats_df.sort_values(["game_id", "is_home"]).reset_index(drop=True)
    
    logger.info(f"Normalized {len(team_stats_df)} team-game stats")
    return team_stats_df


def ingest_nfl_team_stats(
    seasons: Optional[List[int]] = None,
    games_df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Main ingestion function for NFL team stats.
    
    Args:
        seasons: List of seasons to fetch. If None, uses config.
        games_df: Optional DataFrame with games (for game_id matching).
        output_path: Path to save parquet file. If None, uses default.
    
    Returns:
        DataFrame with normalized TeamGameStats schema
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
            / "team_stats.parquet"
        )
    
    # Load games if not provided
    if games_df is None:
        games_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "nfl"
            / "staged"
            / "games.parquet"
        )
        if games_path.exists():
            games_df = pd.read_parquet(games_path)
        else:
            raise FileNotFoundError(f"Games file not found: {games_path}")
    
    # Fetch data
    raw_df = fetch_nflverse_team_stats(seasons)
    
    # Save raw data
    raw_output_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "nfl"
        / "raw"
        / "team_stats.parquet"
    )
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_parquet(raw_output_path, index=False)
    logger.info(f"Saved raw team stats data to {raw_output_path}")
    
    # Normalize
    team_stats_df = normalize_to_team_stats_schema(raw_df, games_df)
    
    # Save staged data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    team_stats_df.to_parquet(output_path, index=False)
    logger.info(f"Saved normalized team stats data to {output_path}")
    
    return team_stats_df


if __name__ == "__main__":
    # Run ingestion
    df = ingest_nfl_team_stats()
    print(f"\nIngested {len(df)} team-game stats")
    print(f"\nGames covered: {df['game_id'].nunique()}")
    print(f"\nSample stats:")
    print(df.head(10))

