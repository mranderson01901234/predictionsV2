"""
Main script to scrape coach data from FootballDB.

Usage:
    python scripts/scrape_coaches.py
"""
import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers.coach_scraper import CoachScraper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Scrape FootballDB coach records')
    
    parser.add_argument('--output-dir', type=str, default='data/raw/footballdb/coaches',
                       help='Output directory for scraped data')
    parser.add_argument('--coach-slug', type=str, help='Scrape specific coach detail page')
    
    args = parser.parse_args()
    
    scraper = CoachScraper(output_dir=args.output_dir)
    
    if args.coach_slug:
        # Scrape specific coach detail
        logger.info(f"Scraping coach detail: {args.coach_slug}")
        coach = scraper.scrape_coach_detail(args.coach_slug)
        if coach:
            logger.info(f"Scraped coach: {coach.get('name', 'Unknown')}")
            logger.info(f"Teams: {coach.get('teams', [])}")
        else:
            logger.error("Failed to scrape coach detail")
    else:
        # Scrape all current coaches
        logger.info("Scraping all current NFL coaches")
        df = scraper.scrape_all_coaches()
        
        if df.empty:
            logger.warning("No coaches scraped")
            return
        
        logger.info(f"\n=== Scraping Complete ===")
        logger.info(f"Total coaches: {len(df)}")
        logger.info(f"\nScraper stats: {scraper.scraper.get_stats()}")
        
        # Print summary
        print("\n=== Coach Summary ===")
        print(df[['name', 'team', 'seasons', 'wins', 'losses', 'win_pct']].to_string(index=False))


if __name__ == "__main__":
    main()

