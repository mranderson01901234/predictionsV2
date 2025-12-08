"""
Main script to run all NFL.com scrapers.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers.injury_scraper import InjuryScraper
from scrapers.player_stats_scraper import PlayerStatsScraper
from scrapers.transaction_scraper import TransactionScraper
from storage.database import NFLDataStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# List of starting QBs to scrape (add more as needed)
QB_SLUGS = [
    'patrick-mahomes', 'josh-allen', 'joe-burrow', 'lamar-jackson',
    'jalen-hurts', 'dak-prescott', 'justin-herbert', 'trevor-lawrence',
    'tua-tagovailoa', 'jared-goff', 'brock-purdy', 'geno-smith',
    'kyler-murray', 'jordan-love', 'c-j-stroud', 'anthony-richardson',
    'derek-carr', 'kirk-cousins', 'matthew-stafford', 'baker-mayfield',
    'sam-darnold', 'russell-wilson', 'aaron-rodgers', 'caleb-williams',
    'jayden-daniels', 'bo-nix', 'drake-maye', 'bryce-young',
]


def scrape_injuries(seasons: list, store: NFLDataStore):
    """Scrape injury reports."""
    logger.info("=== SCRAPING INJURIES ===")
    
    scraper = InjuryScraper()
    all_records = []
    
    for season in seasons:
        logger.info(f"Scraping injuries for {season}")
        records = scraper.scrape_season(season)
        all_records.extend(records)
        logger.info(f"  Found {len(records)} records")
    
    store.save_injuries(all_records)
    logger.info(f"Total injury records: {len(all_records)}")
    
    return scraper.get_stats()


def scrape_player_stats(seasons: list, store: NFLDataStore):
    """Scrape player career and situational stats."""
    logger.info("=== SCRAPING PLAYER STATS ===")
    
    scraper = PlayerStatsScraper()
    all_stats = {}
    
    for slug in QB_SLUGS:
        logger.info(f"Scraping stats for {slug}")
        
        player_stats = {
            'career': scraper.scrape_career_stats(slug),
            'situational': {},
            'game_logs': {},
        }
        
        for season in seasons:
            player_stats['situational'][season] = scraper.scrape_situational_stats(slug, season)
            player_stats['game_logs'][season] = scraper.scrape_game_logs(slug, season)
        
        all_stats[slug] = player_stats
    
    # Save different stat types
    store.save_player_stats(all_stats, 'career')
    store.save_player_stats(all_stats, 'situational')
    store.save_player_stats(all_stats, 'game_logs')
    
    return scraper.get_stats()


def scrape_transactions(seasons: list, store: NFLDataStore):
    """Scrape transactions."""
    logger.info("=== SCRAPING TRANSACTIONS ===")
    
    scraper = TransactionScraper()
    all_transactions = []
    
    for season in seasons:
        logger.info(f"Scraping transactions for {season}")
        transactions = scraper.scrape_season_transactions(season)
        all_transactions.extend(transactions)
    
    store.save_transactions(all_transactions)
    logger.info(f"Total transactions: {len(all_transactions)}")
    
    return scraper.get_stats()


def main():
    parser = argparse.ArgumentParser(description="Scrape NFL.com data")
    parser.add_argument('--start-season', type=int, default=2015)
    parser.add_argument('--end-season', type=int, default=2024)
    parser.add_argument('--injuries', action='store_true', help="Scrape injuries")
    parser.add_argument('--player-stats', action='store_true', help="Scrape player stats")
    parser.add_argument('--transactions', action='store_true', help="Scrape transactions")
    parser.add_argument('--all', action='store_true', help="Scrape everything")
    parser.add_argument('--output-dir', default='data/final', help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(__file__).parent.parent / args.output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    logs_path = Path(__file__).parent.parent / 'logs'
    logs_path.mkdir(exist_ok=True)
    
    # Initialize storage
    store = NFLDataStore(str(output_path))
    
    seasons = list(range(args.start_season, args.end_season + 1))
    logger.info(f"Scraping seasons: {seasons}")
    
    stats = {}
    
    if args.injuries or args.all:
        stats['injuries'] = scrape_injuries(seasons, store)
    
    if args.player_stats or args.all:
        stats['player_stats'] = scrape_player_stats(seasons, store)
    
    if args.transactions or args.all:
        stats['transactions'] = scrape_transactions(seasons, store)
    
    # Print summary
    logger.info("\n=== SCRAPING COMPLETE ===")
    for scraper_name, scraper_stats in stats.items():
        logger.info(f"{scraper_name}: {scraper_stats}")


if __name__ == "__main__":
    main()

