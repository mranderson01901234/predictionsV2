"""
Phase 1A Pipeline Runner

Runs the complete Phase 1A ingestion pipeline:
1. Ingest NFL schedules
2. Ingest NFL odds (from CSV or nflverse)
3. Join games and markets
4. Validate results
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ingestion.nfl.schedule import ingest_nfl_schedules
from ingestion.nfl.odds import ingest_nfl_odds
from ingestion.nfl.join_games_markets import (
    load_games_and_markets,
    join_games_markets,
    save_joined_data,
)
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config():
    """Load NFL data configuration."""
    config_path = Path(__file__).parent.parent.parent / "config" / "data" / "nfl.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_phase1a(odds_csv_path: Path = None):
    """
    Run complete Phase 1A ingestion pipeline.
    
    Args:
        odds_csv_path: Optional path to CSV file with odds data.
                       If None, attempts to fetch from nflverse.
    """
    logger.info("=" * 60)
    logger.info("Starting Phase 1A NFL Ingestion Pipeline")
    logger.info("=" * 60)
    
    config = load_config()
    required_seasons = config["nfl"]["schedule"]["seasons"]
    
    # Step 1: Ingest schedules
    logger.info("\n[Step 1/3] Ingesting NFL schedules...")
    try:
        games_df = ingest_nfl_schedules()
        logger.info(f"✓ Ingested {len(games_df)} games")
        logger.info(f"  Seasons: {games_df['season'].min()} - {games_df['season'].max()}")
    except Exception as e:
        logger.error(f"✗ Failed to ingest schedules: {e}")
        raise
    
    # Step 2: Ingest odds
    logger.info("\n[Step 2/3] Ingesting NFL odds...")
    try:
        if odds_csv_path and Path(odds_csv_path).exists():
            logger.info(f"  Using CSV file: {odds_csv_path}")
            markets_df = ingest_nfl_odds(csv_path=odds_csv_path, games_df=games_df)
        else:
            logger.info("  Attempting to fetch from nflverse...")
            markets_df = ingest_nfl_odds(games_df=games_df)
        
        logger.info(f"✓ Ingested {len(markets_df)} market entries")
        logger.info(f"  Seasons: {markets_df['season'].min()} - {markets_df['season'].max()}")
    except Exception as e:
        logger.error(f"✗ Failed to ingest odds: {e}")
        logger.error("  Hint: Provide a CSV file with odds data or ensure nflverse has odds data")
        logger.error("  Run: python -m ingestion.nfl.generate_odds_template to create a template")
        raise
    
    # Step 3: Join and validate
    logger.info("\n[Step 3/3] Joining games and markets...")
    try:
        joined_df = join_games_markets(
            games_df,
            markets_df,
            validate=True,
            required_seasons=required_seasons,
        )
        
        output_path = save_joined_data(joined_df)
        logger.info(f"✓ Joined {len(joined_df)} games with markets")
        logger.info(f"  Output: {output_path}")
    except Exception as e:
        logger.error(f"✗ Failed to join games and markets: {e}")
        raise
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Phase 1A Pipeline Complete!")
    logger.info("=" * 60)
    logger.info(f"\nOutput files:")
    logger.info(f"  - data/nfl/staged/games.parquet ({len(games_df)} games)")
    logger.info(f"  - data/nfl/staged/markets.parquet ({len(markets_df)} markets)")
    logger.info(f"  - data/nfl/staged/games_markets.parquet ({len(joined_df)} joined)")
    
    # Validation summary
    missing_markets = len(games_df) - len(joined_df)
    if missing_markets > 0:
        logger.warning(f"\n⚠ Warning: {missing_markets} games are missing market data")
    else:
        logger.info("\n✓ All games have matching market data")
    
    return games_df, markets_df, joined_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Phase 1A NFL ingestion pipeline")
    parser.add_argument(
        "--odds-csv",
        type=Path,
        help="Path to CSV file with odds data",
        default=None,
    )
    
    args = parser.parse_args()
    
    try:
        run_phase1a(odds_csv_path=args.odds_csv)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

