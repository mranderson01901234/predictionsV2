"""
Ingest NFL schedule and odds for 2025 Week 14 (DET @ DAL).

Ensures the schedule table includes this game and odds from The Odds API are pulled.
Saves to parquet and refreshes the feature table.
"""

import sys
from pathlib import Path
import pandas as pd
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.nfl.schedule import ingest_nfl_schedules, form_game_id, normalize_team_abbreviation
from ingestion.nfl.odds import ingest_nfl_odds, fetch_odds_api_historical
from ingestion.nfl.join_games_markets import join_games_markets, save_joined_data
from orchestration.pipelines.feature_pipeline import run_baseline_feature_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def update_config_for_2025():
    """Update config to include 2025 season."""
    import yaml
    
    config_path = Path(__file__).parent.parent / "config" / "data" / "nfl.yaml"
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    seasons = config["nfl"]["schedule"]["seasons"]
    if 2025 not in seasons:
        seasons.append(2025)
        config["nfl"]["schedule"]["seasons"] = sorted(seasons)
        
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Updated config to include 2025 season")
        logger.info(f"Seasons in config: {config['nfl']['schedule']['seasons']}")
    else:
        logger.info(f"2025 already in config seasons: {seasons}")


def ensure_game_in_schedule(games_df: pd.DataFrame, season: int, week: int, away_team: str, home_team: str) -> pd.DataFrame:
    """
    Ensure a specific game exists in the schedule.
    
    If the game doesn't exist, add it with minimal required fields.
    """
    game_id = form_game_id(season, week, away_team, home_team)
    
    if game_id in games_df['game_id'].values:
        logger.info(f"Game {game_id} already exists in schedule")
        return games_df
    
    logger.info(f"Game {game_id} not found in schedule. Adding it...")
    
    # Estimate game date (Week 14 is typically early December)
    # For 2025 season, Week 14 would be around December 7-14
    # We'll use December 7 as a placeholder
    game_date = datetime(2025, 12, 7)
    
    # Create new game row
    new_game = pd.DataFrame([{
        'game_id': game_id,
        'season': season,
        'week': week,
        'date': game_date,
        'home_team': normalize_team_abbreviation(home_team),
        'away_team': normalize_team_abbreviation(away_team),
        'home_score': None,
        'away_score': None,
    }])
    
    # Append to games_df
    games_df = pd.concat([games_df, new_game], ignore_index=True)
    
    logger.info(f"Added game {game_id} to schedule")
    return games_df


def fetch_odds_for_specific_game(season: int, week: int, away_team: str, home_team: str) -> pd.DataFrame:
    """
    Fetch odds from The Odds API for a specific game.
    
    Uses the historical endpoint with date range around Week 14.
    """
    logger.info(f"Fetching odds from The Odds API for {away_team} @ {home_team}, {season} Week {week}...")
    
    # Week 14 in 2025 would be around December 7-14
    # Fetch odds for dates around that time
    start_date = datetime(2025, 12, 1)
    end_date = datetime(2025, 12, 15)
    
    # Fetch odds for the season (this will include our game if available)
    odds_df = fetch_odds_api_historical(
        seasons=[season],
        api_key=None,  # Will read from config
        regions="us",
        markets="spreads,totals",
    )
    
    if len(odds_df) == 0:
        logger.warning("No odds data returned from The Odds API")
        return pd.DataFrame()
    
    # Filter for our specific game
    game_id = form_game_id(season, week, away_team, home_team)
    game_odds = odds_df[odds_df['game_id'] == game_id].copy()
    
    if len(game_odds) > 0:
        logger.info(f"Found odds for {game_id}: spread={game_odds.iloc[0].get('close_spread', 'N/A')}, total={game_odds.iloc[0].get('close_total', 'N/A')}")
    else:
        logger.warning(f"Odds not found for {game_id} in API response")
        logger.info(f"Available game_ids in response: {odds_df['game_id'].unique()[:10]}")
    
    return odds_df


def main():
    """Main ingestion workflow."""
    logger.info("=" * 80)
    logger.info("Ingesting NFL Schedule and Odds for 2025 Week 14 (DET @ DAL)")
    logger.info("=" * 80)
    
    season = 2025
    week = 14
    away_team = "DET"
    home_team = "DAL"
    game_id = form_game_id(season, week, away_team, home_team)
    
    project_root = Path(__file__).parent.parent
    games_path = project_root / "data" / "nfl" / "staged" / "games.parquet"
    markets_path = project_root / "data" / "nfl" / "staged" / "markets.parquet"
    games_markets_path = project_root / "data" / "nfl" / "staged" / "games_markets.parquet"
    
    # Step 1: Update config to include 2025
    logger.info("\n[Step 1/5] Updating config for 2025 season...")
    update_config_for_2025()
    
    # Step 2: Ingest schedules (including 2025)
    logger.info("\n[Step 2/5] Ingesting NFL schedules...")
    try:
        games_df = ingest_nfl_schedules(seasons=[2025])
        logger.info(f"✓ Ingested {len(games_df)} games for 2025")
        
        # Check if our game exists
        if game_id in games_df['game_id'].values:
            logger.info(f"✓ Game {game_id} found in schedule")
        else:
            logger.warning(f"Game {game_id} not found in schedule. Adding manually...")
            games_df = ensure_game_in_schedule(games_df, season, week, away_team, home_team)
            
            # Save updated games
            games_path.parent.mkdir(parents=True, exist_ok=True)
            games_df.to_parquet(games_path, index=False)
            logger.info(f"Saved updated games to {games_path}")
    except Exception as e:
        logger.error(f"Error ingesting schedules: {e}")
        logger.info("Attempting to load existing games and add our game...")
        
        if games_path.exists():
            games_df = pd.read_parquet(games_path)
            games_df = ensure_game_in_schedule(games_df, season, week, away_team, home_team)
            games_df.to_parquet(games_path, index=False)
            logger.info(f"Updated games file with {game_id}")
        else:
            raise
    
    # Step 3: Fetch odds from The Odds API
    logger.info("\n[Step 3/5] Fetching odds from The Odds API...")
    try:
        odds_df = fetch_odds_for_specific_game(season, week, away_team, home_team)
        
        if len(odds_df) > 0:
            # Merge with existing markets if they exist
            if markets_path.exists():
                existing_markets = pd.read_parquet(markets_path)
                # Remove duplicates
                odds_df = pd.concat([existing_markets, odds_df]).drop_duplicates(subset=['game_id'], keep='last')
                logger.info(f"Merged with existing markets. Total: {len(odds_df)} games")
            
            # Save markets
            markets_path.parent.mkdir(parents=True, exist_ok=True)
            odds_df.to_parquet(markets_path, index=False)
            logger.info(f"✓ Saved odds to {markets_path}")
            
            # Check if our game has odds
            game_odds = odds_df[odds_df['game_id'] == game_id]
            if len(game_odds) > 0:
                logger.info(f"✓ Odds found for {game_id}")
                logger.info(f"  Close spread: {game_odds.iloc[0].get('close_spread', 'N/A')}")
                logger.info(f"  Close total: {game_odds.iloc[0].get('close_total', 'N/A')}")
            else:
                logger.warning(f"⚠ Odds not found for {game_id} in API response")
        else:
            logger.warning("⚠ No odds data returned from API. Using existing markets if available.")
            if not markets_path.exists():
                logger.warning("⚠ No existing markets file found. Game will have no odds data.")
    except Exception as e:
        logger.error(f"Error fetching odds: {e}")
        logger.warning("Continuing without odds data...")
    
    # Step 4: Join games and markets
    logger.info("\n[Step 4/5] Joining games and markets...")
    try:
        # Load games and markets
        games_df = pd.read_parquet(games_path)
        
        if markets_path.exists():
            markets_df = pd.read_parquet(markets_path)
        else:
            logger.warning("No markets file found. Creating empty markets DataFrame...")
            markets_df = pd.DataFrame(columns=['game_id', 'close_spread', 'close_total', 'open_spread', 'open_total'])
        
        # Join (use left join to keep games without markets)
        # We'll manually join since join_games_markets uses inner join
        if len(markets_df) > 0:
            joined_df = games_df.merge(markets_df, on="game_id", how="left", suffixes=("", "_market"))
            # Drop duplicate columns if any
            if "season_market" in joined_df.columns:
                joined_df = joined_df.drop(columns=["season_market"])
            if "week_market" in joined_df.columns:
                joined_df = joined_df.drop(columns=["week_market"])
        else:
            # No markets, just use games
            joined_df = games_df.copy()
            # Add empty market columns
            joined_df['close_spread'] = None
            joined_df['close_total'] = None
            joined_df['open_spread'] = None
            joined_df['open_total'] = None
        
        # Sort by season, week, date
        joined_df = joined_df.sort_values(["season", "week", "date"]).reset_index(drop=True)
        
        # Save joined data
        save_joined_data(joined_df, games_markets_path)
        logger.info(f"✓ Saved joined data to {games_markets_path}")
        
        # Verify our game
        game_data = joined_df[joined_df['game_id'] == game_id]
        if len(game_data) > 0:
            logger.info(f"✓ Game {game_id} in joined data")
            logger.info(f"  Home: {game_data.iloc[0]['home_team']}, Away: {game_data.iloc[0]['away_team']}")
            logger.info(f"  Spread: {game_data.iloc[0].get('close_spread', 'N/A')}")
            logger.info(f"  Total: {game_data.iloc[0].get('close_total', 'N/A')}")
        else:
            logger.warning(f"⚠ Game {game_id} not found in joined data")
    except Exception as e:
        logger.error(f"Error joining games and markets: {e}")
        raise
    
    # Step 5: Refresh feature table
    logger.info("\n[Step 5/5] Refreshing feature table...")
    try:
        feature_df = run_baseline_feature_pipeline()
        logger.info(f"✓ Refreshed feature table: {len(feature_df)} games")
        
        # Check if our game is in features
        game_features = feature_df[feature_df['game_id'] == game_id]
        if len(game_features) > 0:
            logger.info(f"✓ Game {game_id} in feature table")
            logger.info(f"  Features: {len([c for c in game_features.columns if c.startswith(('home_', 'away_'))])} team features")
        else:
            logger.warning(f"⚠ Game {game_id} not found in feature table")
            logger.info("  This may be normal if team stats are not yet available for 2025")
    except Exception as e:
        logger.error(f"Error refreshing feature table: {e}")
        logger.warning("Feature table refresh failed, but schedule and odds ingestion completed")
    
    logger.info("\n" + "=" * 80)
    logger.info("Ingestion Complete!")
    logger.info("=" * 80)
    logger.info(f"Game ID: {game_id}")
    logger.info(f"Schedule: {'✓' if game_id in games_df['game_id'].values else '✗'}")
    logger.info(f"Odds: {'✓' if markets_path.exists() and len(pd.read_parquet(markets_path)[pd.read_parquet(markets_path)['game_id'] == game_id]) > 0 else '✗'}")
    logger.info(f"Features: {'✓' if 'feature_df' in locals() and len(feature_df[feature_df['game_id'] == game_id]) > 0 else '✗'}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

