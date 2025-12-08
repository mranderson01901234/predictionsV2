"""
Script to scrape injury reports only.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers.injury_scraper import InjuryScraper
from scrapers.api_injury_scraper import APIInjuryScraper
from scrapers.playwright_scraper import PlaywrightInjuryScraper
from storage.database import NFLDataStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Scrape NFL.com injury reports")
    parser.add_argument('--start-season', type=int, default=2020)
    parser.add_argument('--end-season', type=int, default=2024)
    parser.add_argument('--output-dir', default='data/final')
    parser.add_argument('--method', choices=['api', 'playwright', 'selenium', 'auto'], 
                       default='auto', help='Scraping method (auto tries API first, then Playwright)')
    parser.add_argument('--headless', action='store_true', default=True, 
                       help='Run browser in headless mode (Playwright only)')
    
    args = parser.parse_args()
    
    output_path = Path(__file__).parent.parent / args.output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    store = NFLDataStore(str(output_path))
    
    # Choose scraper based on method
    if args.method == 'api':
        logger.info("Using API scraper")
        scraper = APIInjuryScraper()
    elif args.method == 'playwright':
        logger.info("Using Playwright scraper")
        scraper = PlaywrightInjuryScraper(headless=args.headless)
    elif args.method == 'selenium':
        logger.info("Using Selenium scraper")
        scraper = InjuryScraper()  # Will use Selenium variant
    else:  # auto
        logger.info("Using auto method (API → Playwright fallback)")
        scraper = None  # Will try both
    
    seasons = list(range(args.start_season, args.end_season + 1))
    
    all_records = []
    for season in seasons:
        logger.info(f"Scraping injuries for {season}")
        
        if args.method == 'auto':
            # Try API first
            try:
                api_scraper = APIInjuryScraper()
                records = api_scraper.scrape_season(season)
                if records:
                    logger.info(f"  API scraper found {len(records)} records")
                    all_records.extend(records)
                    continue
            except Exception as e:
                logger.warning(f"  API scraper failed: {e}")
            
            # Fallback to Playwright
            try:
                playwright_scraper = PlaywrightInjuryScraper(headless=args.headless)
                records = playwright_scraper.scrape_season(season)
                logger.info(f"  Playwright scraper found {len(records)} records")
                all_records.extend(records)
            except Exception as e:
                logger.error(f"  Playwright scraper failed: {e}")
        else:
            records = scraper.scrape_season(season)
            all_records.extend(records)
    
    store.save_injuries(all_records)
    logger.info(f"Saved {len(all_records)} total injury records")


if __name__ == "__main__":
    main()

