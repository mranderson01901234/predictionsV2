"""
Test the API-based scraper.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_api_scraper():
    """Test the API scraper."""
    logger.info("Testing NFL API Scraper...")
    
    try:
        from scrapers.api_scraper import NFLAPIScraper
        
        scraper = NFLAPIScraper()
        logger.info("✓ API Scraper initialized")
        
        # Test teams endpoint
        logger.info("\nTesting teams endpoint...")
        teams = scraper.get_teams(season=2025)
        
        if teams:
            logger.info(f"✓ Successfully fetched {len(teams)} teams")
            if teams:
                sample_team = teams[0]
                logger.info(f"Sample team: {sample_team.get('name', 'N/A')} ({sample_team.get('abbreviation', 'N/A')})")
        else:
            logger.warning("⚠ No teams data returned (may need fresh auth token)")
        
        # Test injuries endpoint
        logger.info("\nTesting injuries endpoint...")
        injuries = scraper.get_injuries(season=2024, week=1)
        
        if injuries:
            logger.info(f"✓ Successfully fetched injury data")
            logger.info(f"Response type: {type(injuries)}")
            if isinstance(injuries, list):
                logger.info(f"Found {len(injuries)} injury records")
            elif isinstance(injuries, dict):
                logger.info(f"Response keys: {list(injuries.keys())}")
        else:
            logger.warning("⚠ No injury data returned (endpoint may need adjustment)")
        
        logger.info("\n✓ API Scraper test complete")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error testing API scraper: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_api_scraper()

