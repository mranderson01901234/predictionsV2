"""
Join Games and Markets Data

Joins games.parquet and markets.parquet on game_id,
validates completeness and data quality.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_games_and_markets(
    games_path: Optional[Path] = None,
    markets_path: Optional[Path] = None,
) -> tuple:
    """
    Load games and markets DataFrames.
    
    Args:
        games_path: Path to games.parquet. If None, uses default.
        markets_path: Path to markets.parquet. If None, uses default.
    
    Returns:
        Tuple of (games_df, markets_df)
    """
    if games_path is None:
        games_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "nfl"
            / "staged"
            / "games.parquet"
        )
    
    if markets_path is None:
        markets_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "nfl"
            / "staged"
            / "markets.parquet"
        )
    
    logger.info(f"Loading games from {games_path}")
    games_df = pd.read_parquet(games_path)
    
    logger.info(f"Loading markets from {markets_path}")
    markets_df = pd.read_parquet(markets_path)
    
    return games_df, markets_df


def validate_game_ids(games_df: pd.DataFrame) -> Dict[str, bool]:
    """
    Validate game_id format and uniqueness.
    
    Returns:
        Dict with validation results
    """
    results = {}
    
    # Check format: nfl_{season}_{week}_{away}_{home}
    expected_pattern = r"^nfl_\d{4}_\d{2}_[A-Z]{2,3}_[A-Z]{2,3}$"
    valid_format = games_df["game_id"].str.match(expected_pattern).all()
    results["valid_format"] = valid_format
    
    if not valid_format:
        invalid = games_df[~games_df["game_id"].str.match(expected_pattern)]
        logger.error(f"Found {len(invalid)} games with invalid game_id format:")
        logger.error(invalid[["game_id"]].head(10))
    
    # Check for duplicates
    duplicates = games_df["game_id"].duplicated()
    has_duplicates = duplicates.any()
    results["no_duplicates"] = not has_duplicates
    
    if has_duplicates:
        dup_ids = games_df[duplicates]["game_id"].unique()
        logger.error(f"Found {len(dup_ids)} duplicate game_ids:")
        logger.error(dup_ids[:10])
    
    return results


def validate_spread_direction(games_df: pd.DataFrame, markets_df: pd.DataFrame) -> Dict[str, bool]:
    """
    Validate spread direction (favorite should be negative).
    
    Spread is from home team perspective:
    - Negative spread = home team favored
    - Positive spread = away team favored
    
    Returns:
        Dict with validation results
    """
    results = {}
    
    # Join to get scores and spreads
    merged = games_df.merge(markets_df[["game_id", "close_spread"]], on="game_id", how="inner")
    
    # Calculate actual margin (home_score - away_score)
    merged["actual_margin"] = merged["home_score"] - merged["away_score"]
    
    # Check if spread direction makes sense
    # If home team wins by more than spread, spread should be negative (home favored)
    # If away team wins by more than spread, spread should be positive (away favored)
    
    # For games where home won: spread should typically be negative
    home_wins = merged[merged["actual_margin"] > 0]
    if len(home_wins) > 0:
        # Most home favorites should have negative spreads
        home_favored_negative = (home_wins["close_spread"] < 0).sum()
        home_favored_ratio = home_favored_negative / len(home_wins)
        results["home_favorite_spread_sanity"] = home_favored_ratio > 0.3  # At least 30% should be negative
    
    # Check for extreme spreads (sanity check)
    extreme_spreads = merged[abs(merged["close_spread"]) > 20]
    results["no_extreme_spreads"] = len(extreme_spreads) < len(merged) * 0.05  # Less than 5% extreme
    
    if len(extreme_spreads) > 0:
        logger.warning(f"Found {len(extreme_spreads)} games with extreme spreads (>20 points)")
    
    return results


def validate_completeness(
    games_df: pd.DataFrame,
    markets_df: pd.DataFrame,
    required_seasons: List[int] = None,
) -> Dict[str, bool]:
    """
    Validate that all games have matching market entries.
    
    Args:
        games_df: Games DataFrame
        markets_df: Markets DataFrame
        required_seasons: List of seasons that must be present
    
    Returns:
        Dict with validation results
    """
    results = {}
    
    # Check for missing markets
    games_with_markets = games_df.merge(
        markets_df[["game_id"]], on="game_id", how="left", indicator=True
    )
    missing_markets = games_with_markets[games_with_markets["_merge"] == "left_only"]
    
    results["all_games_have_markets"] = len(missing_markets) == 0
    
    if len(missing_markets) > 0:
        logger.error(f"Found {len(missing_markets)} games without market data:")
        logger.error(missing_markets[["game_id", "season", "week", "home_team", "away_team"]].head(10))
    
    # Check for orphaned markets (markets without games)
    markets_with_games = markets_df.merge(
        games_df[["game_id"]], on="game_id", how="left", indicator=True
    )
    orphaned_markets = markets_with_games[markets_with_games["_merge"] == "left_only"]
    
    results["no_orphaned_markets"] = len(orphaned_markets) == 0
    
    if len(orphaned_markets) > 0:
        logger.warning(f"Found {len(orphaned_markets)} market entries without matching games")
    
    # Check season coverage
    if required_seasons:
        games_seasons = set(games_df["season"].unique())
        markets_seasons = set(markets_df["season"].unique())
        
        missing_seasons_games = set(required_seasons) - games_seasons
        missing_seasons_markets = set(required_seasons) - markets_seasons
        
        results["all_seasons_in_games"] = len(missing_seasons_games) == 0
        results["all_seasons_in_markets"] = len(missing_seasons_markets) == 0
        
        if missing_seasons_games:
            logger.error(f"Missing seasons in games: {missing_seasons_games}")
        if missing_seasons_markets:
            logger.error(f"Missing seasons in markets: {missing_seasons_markets}")
    
    return results


def join_games_markets(
    games_df: pd.DataFrame,
    markets_df: pd.DataFrame,
    validate: bool = True,
    required_seasons: List[int] = None,
) -> pd.DataFrame:
    """
    Join games and markets DataFrames.
    
    Args:
        games_df: Games DataFrame
        markets_df: Markets DataFrame
        validate: Whether to run validation checks
        required_seasons: List of seasons that must be present
    
    Returns:
        Joined DataFrame
    """
    logger.info("Joining games and markets data")
    
    if validate:
        logger.info("Running validation checks...")
        
        # Validate game_ids
        game_id_validation = validate_game_ids(games_df)
        for check, passed in game_id_validation.items():
            status = "PASS" if passed else "FAIL"
            logger.info(f"  {check}: {status}")
        
        # Validate completeness
        completeness_validation = validate_completeness(games_df, markets_df, required_seasons)
        for check, passed in completeness_validation.items():
            status = "PASS" if passed else "FAIL"
            logger.info(f"  {check}: {status}")
        
        # Validate spread direction
        spread_validation = validate_spread_direction(games_df, markets_df)
        for check, passed in spread_validation.items():
            status = "PASS" if passed else "FAIL"
            logger.info(f"  {check}: {status}")
        
        # Check if all validations passed
        all_validations = {
            **game_id_validation,
            **completeness_validation,
            **spread_validation,
        }
        all_passed = all(all_validations.values())
        
        if not all_passed:
            logger.warning("Some validations failed. Proceeding with join anyway.")
    
    # Perform join
    # Handle duplicate columns (season, week exist in both)
    # Use suffixes to identify, then keep games_df versions
    joined_df = games_df.merge(markets_df, on="game_id", how="inner", suffixes=("", "_market"))
    
    # If duplicate columns were created, drop the market versions
    if "season_market" in joined_df.columns:
        joined_df = joined_df.drop(columns=["season_market"])
    if "week_market" in joined_df.columns:
        joined_df = joined_df.drop(columns=["week_market"])
    
    # Ensure we have season and week columns
    if "season" not in joined_df.columns:
        # Fallback: use season_x or season_y if they exist
        if "season_x" in joined_df.columns:
            joined_df["season"] = joined_df["season_x"]
            joined_df = joined_df.drop(columns=["season_x", "season_y"] if "season_y" in joined_df.columns else ["season_x"])
        elif "season_y" in joined_df.columns:
            joined_df["season"] = joined_df["season_y"]
            joined_df = joined_df.drop(columns=["season_y"])
    
    if "week" not in joined_df.columns:
        # Fallback: use week_x or week_y if they exist
        if "week_x" in joined_df.columns:
            joined_df["week"] = joined_df["week_x"]
            joined_df = joined_df.drop(columns=["week_x", "week_y"] if "week_y" in joined_df.columns else ["week_x"])
        elif "week_y" in joined_df.columns:
            joined_df["week"] = joined_df["week_y"]
            joined_df = joined_df.drop(columns=["week_y"])
    
    # Sort by season, week, date
    joined_df = joined_df.sort_values(["season", "week", "date"]).reset_index(drop=True)
    
    logger.info(f"Joined {len(joined_df)} games with markets")
    return joined_df


def save_joined_data(
    joined_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Save joined data to parquet file.
    
    Args:
        joined_df: Joined DataFrame
        output_path: Output path. If None, uses default.
    
    Returns:
        Path to saved file
    """
    if output_path is None:
        output_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "nfl"
            / "staged"
            / "games_markets.parquet"
        )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joined_df.to_parquet(output_path, index=False)
    logger.info(f"Saved joined data to {output_path}")
    
    return output_path


if __name__ == "__main__":
    # Load data
    games_df, markets_df = load_games_and_markets()
    
    # Required seasons from config
    import yaml
    config_path = Path(__file__).parent.parent.parent / "config" / "data" / "nfl.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    required_seasons = config["nfl"]["schedule"]["seasons"]
    
    # Join and validate
    joined_df = join_games_markets(
        games_df,
        markets_df,
        validate=True,
        required_seasons=required_seasons,
    )
    
    # Save
    output_path = save_joined_data(joined_df)
    
    print(f"\nJoined {len(joined_df)} games with markets")
    print(f"\nSeasons: {joined_df['season'].min()} - {joined_df['season'].max()}")
    print(f"\nSample joined data:")
    print(joined_df.head(10))

