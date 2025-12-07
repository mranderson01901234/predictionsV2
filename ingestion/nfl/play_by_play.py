"""
NFL Play-by-Play Data Ingestion Module

Fetches NFL play-by-play data from nflverse/nflfastR sources
and normalizes to the core PlayByPlay schema for EPA/SR calculation.
"""

import pandas as pd
import yaml
from pathlib import Path
from typing import List, Optional
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
    
    Reuses the same normalization logic from schedule.py.
    
    Args:
        team: Team abbreviation to normalize
        season: Optional season year for context
    """
    from ingestion.nfl.schedule import normalize_team_abbreviation as normalize_team
    return normalize_team(team, season)


def form_game_id(season: int, week: int, away_team: str, home_team: str) -> str:
    """
    Form game_id in format: nfl_{season}_{week}_{away}_{home}
    
    Reuses the same logic from schedule.py.
    """
    from ingestion.nfl.schedule import form_game_id
    return form_game_id(season, week, away_team, home_team)


def fetch_nflverse_pbp(seasons: List[int]) -> pd.DataFrame:
    """
    Fetch NFL play-by-play data from nflverse.
    
    Args:
        seasons: List of season years to fetch
    
    Returns:
        DataFrame with play-by-play data
    """
    try:
        import nfl_data_py as nfl
    except ImportError:
        raise ImportError(
            "nfl_data_py (nflverse) is required. Install with: pip install nfl-data-py"
        )
    
    logger.info(f"Fetching NFL play-by-play data for seasons {seasons}")
    
    all_pbp = []
    
    for season in seasons:
        try:
            logger.info(f"Fetching PBP data for season {season}...")
            # Fetch all columns - nflverse will return what's available
            pbp = nfl.import_pbp_data(
                [season],
                columns=None,  # Get all columns
                downcast=True,
                cache=False,
            )
            
            # Validate that critical columns exist
            # nflverse uses 'old_game_id' for game identifier
            if "old_game_id" not in pbp.columns and "game_id" not in pbp.columns:
                raise ValueError(f"Missing game_id column in season {season} PBP data")
            
            if "epa" not in pbp.columns:
                raise ValueError(f"Missing epa column in season {season} PBP data")
            
            if "posteam" not in pbp.columns:
                raise ValueError(f"Missing posteam column in season {season} PBP data")
            
            if len(pbp) > 0:
                all_pbp.append(pbp)
                logger.info(f"Fetched {len(pbp)} plays for season {season}")
            else:
                logger.warning(f"No plays found for season {season}")
                
        except Exception as e:
            logger.error(f"Error fetching season {season}: {e}")
            raise
    
    if not all_pbp:
        raise ValueError("No play-by-play data fetched")
    
    df = pd.concat(all_pbp, ignore_index=True)
    logger.info(f"Total plays fetched: {len(df)}")
    return df


def map_nflverse_game_id_to_our_format(
    pbp_df: pd.DataFrame, games_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Map nflverse game_id (old_game_id) to our game_id format.
    
    nflverse uses old_game_id format like: "2015_01_KC_HOU"
    We use format: "nfl_2015_01_KC_HOU"
    
    Args:
        pbp_df: Play-by-play DataFrame with nflverse game_id
        games_df: Optional games DataFrame to validate against
    
    Returns:
        DataFrame with our game_id format
    """
    logger.info("Mapping nflverse game_id to our format")
    
    # nflverse uses 'old_game_id' or 'game_id' column
    # Format is typically: "{season}_{week:02d}_{away}_{home}"
    game_id_col = None
    for col in ["old_game_id", "game_id"]:
        if col in pbp_df.columns:
            game_id_col = col
            break
    
    if game_id_col is None:
        raise ValueError("No game_id column found in PBP data")
    
    # Extract components from nflverse game_id
    def parse_game_id(game_id_str):
        """Parse nflverse game_id format: {season}_{week}_{away}_{home}"""
        if pd.isna(game_id_str):
            return None, None, None, None
        
        parts = str(game_id_str).split("_")
        if len(parts) >= 4:
            season = int(parts[0])
            week = int(parts[1])
            away = parts[2]
            home = parts[3]
            return season, week, away, home
        return None, None, None, None
    
    # Parse game_id components
    parsed = pbp_df[game_id_col].apply(parse_game_id)
    pbp_df["_season_parsed"] = [p[0] for p in parsed]
    pbp_df["_week_parsed"] = [p[1] for p in parsed]
    pbp_df["_away_parsed"] = [p[2] for p in parsed]
    pbp_df["_home_parsed"] = [p[3] for p in parsed]
    
    # Normalize team abbreviations
    pbp_df["_away_norm"] = pbp_df.apply(
        lambda row: normalize_team_abbreviation(row["_away_parsed"], row.get("season")),
        axis=1,
    )
    pbp_df["_home_norm"] = pbp_df.apply(
        lambda row: normalize_team_abbreviation(row["_home_parsed"], row.get("season")),
        axis=1,
    )
    
    # Form our game_id
    pbp_df["game_id"] = pbp_df.apply(
        lambda row: form_game_id(
            row["_season_parsed"] if pd.notna(row["_season_parsed"]) else row.get("season"),
            row["_week_parsed"] if pd.notna(row["_week_parsed"]) else row.get("week"),
            row["_away_norm"],
            row["_home_norm"],
        ),
        axis=1,
    )
    
    # Clean up temporary columns
    pbp_df = pbp_df.drop(
        columns=["_season_parsed", "_week_parsed", "_away_parsed", "_home_parsed", "_away_norm", "_home_norm"],
        errors="ignore",
    )
    
    # Validate against games.parquet if provided
    if games_df is not None:
        games_game_ids = set(games_df["game_id"].unique())
        pbp_game_ids = set(pbp_df["game_id"].unique())
        
        missing_in_pbp = games_game_ids - pbp_game_ids
        if missing_in_pbp:
            logger.warning(f"{len(missing_in_pbp)} games in games.parquet missing from PBP data")
            logger.debug(f"Sample missing games: {list(missing_in_pbp)[:5]}")
        
        extra_in_pbp = pbp_game_ids - games_game_ids
        if extra_in_pbp:
            logger.warning(f"{len(extra_in_pbp)} games in PBP data not in games.parquet")
            logger.debug(f"Sample extra games: {list(extra_in_pbp)[:5]}")
    
    logger.info(f"Mapped {len(pbp_df)} plays to our game_id format")
    return pbp_df


def normalize_to_play_schema(
    pbp_df: pd.DataFrame, games_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Normalize nflverse play-by-play data to PlayByPlay schema.
    
    Required fields:
    - game_id
    - play_id
    - posteam (team with possession)
    - defteam (defending team)
    - play_type
    - epa
    - success (compute if missing)
    - pass/run indicator
    - down
    - ydstogo
    - yardline_100
    - qtr
    - half_seconds_remaining
    
    Args:
        pbp_df: Raw play-by-play DataFrame from nflverse
        games_df: Optional games DataFrame for validation
    
    Returns:
        DataFrame with normalized PlayByPlay schema
    """
    logger.info("Normalizing play-by-play data to PlayByPlay schema")
    
    # Map game_id first
    pbp_df = map_nflverse_game_id_to_our_format(pbp_df, games_df)
    
    # Normalize team abbreviations for posteam and defteam
    pbp_df["posteam_norm"] = pbp_df.apply(
        lambda row: normalize_team_abbreviation(row.get("posteam", ""), row.get("season")),
        axis=1,
    )
    pbp_df["defteam_norm"] = pbp_df.apply(
        lambda row: normalize_team_abbreviation(row.get("defteam", ""), row.get("season")),
        axis=1,
    )
    
    # Compute success if missing
    if "success" not in pbp_df.columns or pbp_df["success"].isna().any():
        logger.info("Computing success indicator from EPA")
        # Success is typically defined as EPA > 0 for most plays
        # But nflverse should have this, so this is a fallback
        pbp_df["success"] = (pbp_df["epa"] > 0).astype(int)
    
    # Ensure pass/run indicators exist
    if "pass" not in pbp_df.columns:
        pbp_df["pass"] = (pbp_df["play_type"] == "pass").astype(int)
    if "rush" not in pbp_df.columns:
        pbp_df["rush"] = (pbp_df["play_type"] == "run").astype(int)
    
    # Build normalized PlayByPlay schema DataFrame
    play_df = pd.DataFrame({
        "game_id": pbp_df["game_id"],
        "play_id": pbp_df.get("play_id", pbp_df.index),
        "posteam": pbp_df["posteam_norm"],
        "defteam": pbp_df["defteam_norm"],
        "play_type": pbp_df.get("play_type", ""),
        "epa": pd.to_numeric(pbp_df.get("epa", 0), errors="coerce").fillna(0.0),
        "success": pd.to_numeric(pbp_df.get("success", 0), errors="coerce").fillna(0).astype(int),
        "is_pass": pbp_df["pass"].fillna(0).astype(int),
        "is_run": pbp_df["rush"].fillna(0).astype(int),
        "down": pd.to_numeric(pbp_df.get("down", 0), errors="coerce").fillna(0).astype(int),
        "ydstogo": pd.to_numeric(pbp_df.get("ydstogo", 0), errors="coerce").fillna(0).astype(int),
        "yardline_100": pd.to_numeric(pbp_df.get("yardline_100", 0), errors="coerce").fillna(0).astype(int),
        "qtr": pd.to_numeric(pbp_df.get("qtr", 0), errors="coerce").fillna(0).astype(int),
        "half_seconds_remaining": pd.to_numeric(
            pbp_df.get("half_seconds_remaining", 0), errors="coerce"
        ).fillna(0).astype(int),
    })
    
    # Filter out invalid plays (no posteam/defteam, no EPA)
    initial_len = len(play_df)
    play_df = play_df[
        (play_df["posteam"].notna()) &
        (play_df["defteam"].notna()) &
        (play_df["posteam"] != "") &
        (play_df["defteam"] != "")
    ]
    if len(play_df) < initial_len:
        logger.warning(f"Dropped {initial_len - len(play_df)} plays with missing posteam/defteam")
    
    # Sort by game_id, play_id
    play_df = play_df.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    
    logger.info(f"Normalized {len(play_df)} plays")
    return play_df


def ingest_nfl_play_by_play(
    seasons: Optional[List[int]] = None,
    games_df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Main ingestion function for NFL play-by-play data.
    
    Args:
        seasons: List of seasons to fetch. If None, uses config.
        games_df: Optional games DataFrame for validation
        output_path: Path to save parquet file. If None, uses default.
    
    Returns:
        DataFrame with normalized PlayByPlay schema
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
            / "plays.parquet"
        )
    
    # Load games for validation if not provided
    if games_df is None:
        games_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "nfl"
            / "staged"
            / "games.parquet"
        )
        if games_path.exists():
            logger.info(f"Loading games from {games_path} for validation")
            games_df = pd.read_parquet(games_path)
        else:
            logger.warning("games.parquet not found, skipping validation")
            games_df = None
    
    # Fetch data
    raw_df = fetch_nflverse_pbp(seasons)
    
    # Save raw data
    raw_output_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "nfl"
        / "raw"
        / "play_by_play.parquet"
    )
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_parquet(raw_output_path, index=False)
    logger.info(f"Saved raw play-by-play data to {raw_output_path}")
    
    # Normalize
    play_df = normalize_to_play_schema(raw_df, games_df)
    
    # Validate: each game_id in games.parquet should appear in plays.parquet
    if games_df is not None:
        games_game_ids = set(games_df["game_id"].unique())
        plays_game_ids = set(play_df["game_id"].unique())
        
        missing_games = games_game_ids - plays_game_ids
        if missing_games:
            logger.warning(
                f"{len(missing_games)} games in games.parquet missing from plays.parquet"
            )
            logger.debug(f"Sample missing games: {list(missing_games)[:5]}")
        else:
            logger.info("✓ All games in games.parquet have plays")
    
    # Save staged data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    play_df.to_parquet(output_path, index=False)
    logger.info(f"Saved normalized play-by-play data to {output_path}")
    
    return play_df


if __name__ == "__main__":
    # Run ingestion
    df = ingest_nfl_play_by_play()
    print(f"\nIngested {len(df)} plays")
    print(f"\nGames covered: {df['game_id'].nunique()}")
    print(f"\nSeasons: {df['game_id'].str.split('_').str[1].astype(int).min()} - {df['game_id'].str.split('_').str[1].astype(int).max()}")
    print(f"\nSample plays:")
    print(df.head(10))

