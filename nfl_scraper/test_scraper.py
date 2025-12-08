"""
Quick test script to verify the scraper works.
"""

import sys
import logging
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_base_scraper():
    """Test the base scraper can fetch a simple page."""
    logger.info("Testing BaseScraper...")
    
    try:
        from scrapers.base_scraper import BaseScraper
        
        scraper = BaseScraper()
        logger.info("✓ BaseScraper initialized")
        
        # Test fetching a simple page (NFL.com homepage)
        logger.info("Testing fetch of NFL.com homepage...")
        html = scraper.fetch("https://www.nfl.com/", use_cache=False)
        
        if html and len(html) > 1000:
            logger.info(f"✓ Successfully fetched NFL.com (got {len(html)} bytes)")
            return True
        else:
            logger.error("✗ Failed to fetch NFL.com or got empty response")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error testing BaseScraper: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_injury_scraper():
    """Test the injury scraper."""
    logger.info("\nTesting InjuryScraper...")
    
    try:
        from scrapers.injury_scraper import InjuryScraper
        
        scraper = InjuryScraper()
        logger.info("✓ InjuryScraper initialized")
        
        # Try to fetch current week injuries (2024 season, week 1)
        logger.info("Testing fetch of injury report (2024, week 1)...")
        records = scraper.scrape_week(2024, 1, use_cache=False)
        
        logger.info(f"✓ Scraped {len(records)} injury records")
        
        if records:
            logger.info(f"Sample record: {records[0].player_name} - {records[0].injury_types}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error testing InjuryScraper: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parser():
    """Test the injury parser with sample HTML."""
    logger.info("\nTesting InjuryParser...")
    
    try:
        from parsers.injury_parser import InjuryParser
        
        parser = InjuryParser()
        logger.info("✓ InjuryParser initialized")
        
        # Test parsing functions
        injuries, is_resting = parser._parse_injury_types("Hamstring")
        assert injuries == ["Hamstring"]
        assert not is_resting
        logger.info("✓ Injury type parsing works")
        
        status = parser._normalize_practice_status("Full Participation in Practice")
        assert status == "Full"
        logger.info("✓ Practice status normalization works")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error testing parser: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_storage():
    """Test the storage layer."""
    logger.info("\nTesting NFLDataStore...")
    
    try:
        from storage.database import NFLDataStore
        from storage.schemas import InjuryRecord
        from datetime import date
        
        store = NFLDataStore("data/final")
        logger.info("✓ NFLDataStore initialized")
        
        # Create a test record
        test_record = InjuryRecord(
            season=2024,
            week=1,
            game_date=date(2024, 9, 8),
            team="KC",
            opponent="BAL",
            player_name="Test Player",
            player_id="test-player",
            position="QB",
            injury_type="Hamstring",
            injury_types=["Hamstring"],
            practice_status_wed=None,
            practice_status_thu=None,
            practice_status_fri=None,
            practice_status_final="Limited",
            game_status="Questionable",
            is_resting=False,
        )
        
        # Save test record
        store.save_injuries([test_record], "test_injuries.parquet")
        logger.info("✓ Saved test injury record")
        
        # Load it back
        df = store.load_injuries("test_injuries.parquet")
        assert len(df) == 1
        assert df.iloc[0]['player_name'] == "Test Player"
        logger.info("✓ Loaded test injury record")
        
        # Clean up
        test_file = Path("data/final/test_injuries.parquet")
        if test_file.exists():
            test_file.unlink()
        logger.info("✓ Cleaned up test file")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error testing storage: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("NFL Scraper Test Suite")
    logger.info("=" * 60)
    
    results = []
    
    # Test parser first (no network required)
    results.append(("Parser", test_parser()))
    
    # Test storage
    results.append(("Storage", test_storage()))
    
    # Test base scraper (requires network)
    results.append(("BaseScraper", test_base_scraper()))
    
    # Test injury scraper (requires network, may fail if HTML structure changed)
    results.append(("InjuryScraper", test_injury_scraper()))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        logger.info("\n✓ All tests passed!")
    else:
        logger.info("\n✗ Some tests failed. Check logs above for details.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

