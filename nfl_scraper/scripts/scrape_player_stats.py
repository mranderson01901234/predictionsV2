"""
Script to scrape player stats only.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers.player_stats_scraper import PlayerStatsScraper
from storage.database import NFLDataStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QB_SLUGS = [
    'patrick-mahomes', 'josh-allen', 'joe-burrow', 'lamar-jackson',
    'jalen-hurts', 'dak-prescott', 'justin-herbert', 'trevor-lawrence',
    'tua-tagovailoa', 'jared-goff', 'brock-purdy', 'geno-smith',
]


def main():
    parser = argparse.ArgumentParser(description="Scrape NFL.com player stats")
    parser.add_argument('--start-season', type=int, default=2020)
    parser.add_argument('--end-season', type=int, default=2024)
    parser.add_argument('--output-dir', default='data/final')
    parser.add_argument('--players', nargs='+', help="Player slugs to scrape")
    
    args = parser.parse_args()
    
    output_path = Path(__file__).parent.parent / args.output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    store = NFLDataStore(str(output_path))
    scraper = PlayerStatsScraper()
    
    seasons = list(range(args.start_season, args.end_season + 1))
    players = args.players or QB_SLUGS
    
    all_stats = {}
    for slug in players:
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
    
    store.save_player_stats(all_stats, 'career')
    store.save_player_stats(all_stats, 'situational')
    store.save_player_stats(all_stats, 'game_logs')
    
    logger.info("Player stats scraping complete")


if __name__ == "__main__":
    main()

