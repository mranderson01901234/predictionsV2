"""
Main script to scrape player splits from FootballDB.

Usage:
    python scripts/scrape_splits.py --year 2024 --priority-only
    python scripts/scrape_splits.py --start-year 2015 --end-year 2024 --priority-only
    python scripts/scrape_splits.py --surface-only --start-year 2015 --end-year 2024
"""
import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers.player_splits_scraper import PlayerSplitsScraper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Scrape FootballDB player splits')
    
    # Year options
    parser.add_argument('--year', type=int, help='Single year to scrape')
    parser.add_argument('--start-year', type=int, default=2015, help='Start year for historical scraping')
    parser.add_argument('--end-year', type=int, default=2024, help='End year for historical scraping')
    
    # Scraping options
    parser.add_argument('--priority-only', action='store_true', help='Only scrape priority splits')
    parser.add_argument('--surface-only', action='store_true', help='Only scrape surface splits (grass/turf)')
    parser.add_argument('--all-splits', action='store_true', help='Scrape all available splits')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='data/raw/footballdb/player_splits',
                       help='Output directory for scraped data')
    
    args = parser.parse_args()
    
    scraper = PlayerSplitsScraper(output_dir=args.output_dir)
    
    if args.surface_only:
        # Surface splits only (highest value)
        logger.info("Scraping surface splits only (Grass vs Turf)")
        df = scraper.scrape_surface_splits_only(
            start_year=args.start_year,
            end_year=args.end_year
        )
        
    elif args.year:
        # Single year
        logger.info(f"Scraping year {args.year}")
        df = scraper.scrape_season(
            year=args.year,
            priority_only=args.priority_only and not args.all_splits
        )
        
    else:
        # Historical scraping
        logger.info(f"Scraping historical data: {args.start_year} to {args.end_year}")
        df = scraper.scrape_historical(
            start_year=args.start_year,
            end_year=args.end_year,
            priority_only=args.priority_only and not args.all_splits
        )
    
    if df.empty:
        logger.warning("No data scraped")
        return
    
    logger.info(f"\n=== Scraping Complete ===")
    logger.info(f"Total records: {len(df)}")
    logger.info(f"Stat types: {df['stat_type'].unique().tolist()}")
    logger.info(f"Split types: {df['split_type'].unique().tolist()}")
    logger.info(f"Seasons: {df['season'].unique().tolist()}")
    logger.info(f"\nScraper stats: {scraper.scraper.get_stats()}")


if __name__ == "__main__":
    main()

