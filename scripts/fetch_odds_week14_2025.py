"""
Fetch odds for 2025 Week 14 game (DET vs DAL) from The Odds API.

The game was played on December 4, 2025 (Thursday).
Game ID: nfl_2025_14_DAL_DET (DET home, DAL away)
Score: DET 44, DAL 30
"""

import sys
from pathlib import Path
import pandas as pd
import logging
from datetime import datetime
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.nfl.odds import load_config, map_odds_api_team_to_abbreviation, form_game_id, _parse_odds_api_game
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def fetch_odds_for_date(date_str: str, api_key: str, regions: str = "us", markets: str = "spreads,totals"):
    """
    Fetch odds from The Odds API for a specific date.
    
    Args:
        date_str: Date in YYYY-MM-DD format
        api_key: The Odds API key
        regions: Comma-separated regions
        markets: Comma-separated markets
    
    Returns:
        List of odds entries
    """
    base_url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds-history"
    
    url = f"{base_url}"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "date": date_str,
    }
    
    try:
        logger.info(f"Fetching odds for {date_str}...")
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 429:
            remaining = response.headers.get("x-requests-remaining", "0")
            logger.warning(f"⚠️ Rate limit exceeded. Remaining requests: {remaining}")
            return []
        
        response.raise_for_status()
        data = response.json()
        
        # Check remaining requests
        remaining = response.headers.get("x-requests-remaining")
        if remaining:
            remaining_int = int(remaining) if remaining.isdigit() else 0
            logger.info(f"API requests remaining: {remaining_int}")
        
        all_odds = []
        
        # Process response data
        if isinstance(data, list) and len(data) > 0:
            logger.info(f"Found {len(data)} games for {date_str}")
            for game in data:
                odds_entry = _parse_odds_api_game(game, 2025)
                if odds_entry:
                    all_odds.append(odds_entry)
        elif isinstance(data, dict) and "data" in data:
            logger.info(f"Found {len(data.get('data', []))} games for {date_str}")
            for game in data.get("data", []):
                odds_entry = _parse_odds_api_game(game, 2025)
                if odds_entry:
                    all_odds.append(odds_entry)
        else:
            logger.warning(f"No games found for {date_str}")
        
        return all_odds
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching odds for {date_str}: {e}")
        return []


def main():
    """Main function."""
    logger.info("=" * 80)
    logger.info("Fetching Odds for 2025 Week 14 (DET vs DAL)")
    logger.info("=" * 80)
    
    # Game details
    game_id = "nfl_2025_14_DAL_DET"  # DET home, DAL away
    game_date = "2025-12-04"  # Thursday, December 4, 2025
    
    # Load config
    config = load_config()
    odds_config = config.get("nfl", {}).get("odds", {})
    api_config = odds_config.get("the_odds_api", {})
    api_key = api_config.get("api_key")
    
    if not api_key:
        logger.error("The Odds API key not found in config")
        return
    
    logger.info(f"API key: {api_key[:8]}...{api_key[-4:]}")
    
    # Fetch odds for the game date
    odds_entries = fetch_odds_for_date(
        game_date,
        api_key,
        regions=api_config.get("regions", "us"),
        markets=api_config.get("markets", "spreads,totals"),
    )
    
    if not odds_entries:
        logger.warning("No odds data retrieved")
        return
    
    # Convert to DataFrame
    odds_df = pd.DataFrame(odds_entries)
    logger.info(f"Retrieved odds for {len(odds_df)} games")
    
    # Check if our game is in the results
    game_odds = odds_df[odds_df['game_id'] == game_id]
    if len(game_odds) > 0:
        logger.info(f"✓ Found odds for {game_id}")
        logger.info(f"  Close spread: {game_odds.iloc[0].get('close_spread', 'N/A')}")
        logger.info(f"  Close total: {game_odds.iloc[0].get('close_total', 'N/A')}")
        logger.info(f"  Open spread: {game_odds.iloc[0].get('open_spread', 'N/A')}")
        logger.info(f"  Open total: {game_odds.iloc[0].get('open_total', 'N/A')}")
    else:
        logger.warning(f"⚠ Odds not found for {game_id}")
        logger.info(f"Available game_ids: {odds_df['game_id'].unique()[:10]}")
    
    # Update markets file
    project_root = Path(__file__).parent.parent
    markets_path = project_root / "data" / "nfl" / "staged" / "markets.parquet"
    
    if markets_path.exists():
        existing_markets = pd.read_parquet(markets_path)
        logger.info(f"Loaded existing markets: {len(existing_markets)} games")
        
        # Merge new odds (replace existing entries for same game_id)
        # Remove existing entries for games in odds_df
        existing_markets = existing_markets[~existing_markets['game_id'].isin(odds_df['game_id'])]
        
        # Combine
        updated_markets = pd.concat([existing_markets, odds_df], ignore_index=True)
        logger.info(f"Updated markets: {len(updated_markets)} games")
    else:
        updated_markets = odds_df
        logger.info(f"Created new markets file: {len(updated_markets)} games")
    
    # Save
    markets_path.parent.mkdir(parents=True, exist_ok=True)
    updated_markets.to_parquet(markets_path, index=False)
    logger.info(f"✓ Saved markets to {markets_path}")
    
    # Update games_markets
    logger.info("\nUpdating games_markets.parquet...")
    games_path = project_root / "data" / "nfl" / "staged" / "games.parquet"
    games_markets_path = project_root / "data" / "nfl" / "staged" / "games_markets.parquet"
    
    games_df = pd.read_parquet(games_path)
    
    # Join games with updated markets
    joined_df = games_df.merge(
        updated_markets,
        on="game_id",
        how="left",
        suffixes=("", "_market")
    )
    
    # Drop duplicate columns
    if "season_market" in joined_df.columns:
        joined_df = joined_df.drop(columns=["season_market"])
    if "week_market" in joined_df.columns:
        joined_df = joined_df.drop(columns=["week_market"])
    
    # Sort
    joined_df = joined_df.sort_values(["season", "week", "date"]).reset_index(drop=True)
    
    # Save
    joined_df.to_parquet(games_markets_path, index=False)
    logger.info(f"✓ Saved joined data to {games_markets_path}")
    
    # Verify our game
    game_data = joined_df[joined_df['game_id'] == game_id]
    if len(game_data) > 0:
        logger.info(f"\n✓ Game {game_id} in joined data:")
        logger.info(f"  Home: {game_data.iloc[0]['home_team']}, Away: {game_data.iloc[0]['away_team']}")
        logger.info(f"  Score: {game_data.iloc[0]['home_team']} {game_data.iloc[0].get('home_score', 'N/A')} - {game_data.iloc[0].get('away_score', 'N/A')} {game_data.iloc[0]['away_team']}")
        logger.info(f"  Close spread: {game_data.iloc[0].get('close_spread', 'N/A')}")
        logger.info(f"  Close total: {game_data.iloc[0].get('close_total', 'N/A')}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Odds Fetch Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

