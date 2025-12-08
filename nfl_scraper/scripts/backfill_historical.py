"""
Script to backfill historical data (2015-2024).
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers.injury_scraper import InjuryScraper
from scrapers.transaction_scraper import TransactionScraper
from storage.database import NFLDataStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Backfill historical NFL.com data")
    parser.add_argument('--start-season', type=int, default=2015)
    parser.add_argument('--end-season', type=int, default=2024)
    parser.add_argument('--output-dir', default='data/final')
    parser.add_argument('--injuries', action='store_true')
    parser.add_argument('--transactions', action='store_true')
    parser.add_argument('--all', action='store_true')
    
    args = parser.parse_args()
    
    output_path = Path(__file__).parent.parent / args.output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    store = NFLDataStore(str(output_path))
    seasons = list(range(args.start_season, args.end_season + 1))
    
    if args.injuries or args.all:
        logger.info("Backfilling injuries...")
        scraper = InjuryScraper()
        all_records = scraper.scrape_historical(args.start_season, args.end_season)
        store.save_injuries(all_records)
        logger.info(f"Saved {len(all_records)} injury records")
    
    if args.transactions or args.all:
        logger.info("Backfilling transactions...")
        scraper = TransactionScraper()
        all_transactions = []
        for season in seasons:
            transactions = scraper.scrape_season_transactions(season)
            all_transactions.extend(transactions)
        store.save_transactions(all_transactions)
        logger.info(f"Saved {len(all_transactions)} transaction records")


if __name__ == "__main__":
    main()

