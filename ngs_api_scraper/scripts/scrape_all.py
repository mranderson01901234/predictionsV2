"""
Master script to scrape all NGS data.

Usage:
    # Full historical scrape (2018-2024)
    python scrape_all.py --historical
    
    # Current season only
    python scrape_all.py --season 2024
    
    # Current week update
    python scrape_all.py --current-week
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
import sys
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers.statboard_scraper import StatboardScraper
from scrapers.base_client import NGSClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Scrape NFL Next Gen Stats")
    parser.add_argument('--historical', action='store_true',
                        help="Scrape all historical data (2018-present)")
    parser.add_argument('--season', type=int, default=None,
                        help="Scrape specific season")
    parser.add_argument('--current-week', action='store_true',
                        help="Scrape only current week")
    parser.add_argument('--no-weekly', action='store_true',
                        help="Skip week-by-week data, only season totals")
    parser.add_argument('--output-dir', default='data/raw/ngs',
                        help="Output directory")
    
    args = parser.parse_args()
    
    scraper = StatboardScraper(output_dir=args.output_dir)
    
    if args.historical:
        logger.info("=== Starting Historical Scrape ===")
        data = scraper.scrape_historical(
            start_season=2018,
            include_weekly=not args.no_weekly
        )
        
        for stat_type, df in data.items():
            logger.info(f"{stat_type}: {len(df)} total records")
    
    elif args.season:
        logger.info(f"=== Scraping Season {args.season} ===")
        data = scraper.scrape_season(
            args.season,
            'REG',
            include_weekly=not args.no_weekly
        )
        
        # Also scrape postseason if past January
        if datetime.now().month >= 1:
            post_data = scraper.scrape_season(
                args.season,
                'POST',
                include_weekly=not args.no_weekly
            )
            for k, v in post_data.items():
                if k in data:
                    data[k] = pd.concat([data[k], v], ignore_index=True)
                else:
                    data[k] = v
    
    elif args.current_week:
        logger.info("=== Scraping Current Week ===")
        # Determine current week
        # (Would need schedule data to determine actual current week)
        season = datetime.now().year
        data = scraper.scrape_season(season, 'REG', include_weekly=True)
    
    else:
        # Default: current season
        season = datetime.now().year
        logger.info(f"=== Scraping Current Season {season} ===")
        data = scraper.scrape_season(season, 'REG', include_weekly=True)
    
    logger.info("\n=== Scrape Complete ===")
    logger.info(f"Client stats: {scraper.client.get_stats()}")


if __name__ == "__main__":
    main()

