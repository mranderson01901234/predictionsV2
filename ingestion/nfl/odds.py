"""
NFL Odds Ingestion Module

Fetches historical NFL betting odds (spreads, totals) from free sources
and normalizes to the MarketSnapshot schema.
"""

import pandas as pd
import yaml
from pathlib import Path
from typing import List, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    
    Args:
        seasons: List of seasons to fetch. If None, uses config.
        csv_path: Optional path to CSV file with odds data.
        games_df: Optional DataFrame with games (for validation/matching).
        output_path: Path to save parquet file. If None, uses default.
    
    Returns:
        DataFrame with normalized MarketSnapshot schema
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
            / "markets.parquet"
        )
    
    # Load odds data
    if csv_path and csv_path.exists():
        odds_df = load_odds_from_csv(csv_path)
    else:
        # Try nflverse (placeholder)
        odds_df = fetch_nflverse_odds(seasons)
        
        if len(odds_df) == 0:
            logger.error(
                "No odds data source available. "
                "Please provide a CSV file with historical odds data. "
                "See docs/data_sources.md for data source information."
            )
            raise ValueError("No odds data available")
    
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
    logger.info(f"Saved raw odds data to {raw_output_path}")
    
    # Normalize
    market_df = normalize_to_market_schema(odds_df, games_df)
    
    # Save staged data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    market_df.to_parquet(output_path, index=False)
    logger.info(f"Saved normalized odds data to {output_path}")
    
    return market_df


if __name__ == "__main__":
    # Example usage - requires CSV file or games_df
    import sys
    
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
        df = ingest_nfl_odds(csv_path=csv_path)
    else:
        print("Usage: python odds.py <path_to_odds_csv>")
        print("Or use ingest_nfl_odds() with games_df parameter")

