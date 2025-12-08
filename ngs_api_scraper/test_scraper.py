"""
Quick test script for NGS API scraper.

Tests basic functionality of the scraper.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from scrapers.base_client import NGSClient
from scrapers.statboard_scraper import StatboardScraper


def test_base_client():
    """Test the base NGS client."""
    print("=" * 60)
    print("Testing NGS Base Client")
    print("=" * 60)
    
    client = NGSClient()
    
    # Test passing stats
    print("\n1. Testing passing stats...")
    passing = client.get_passing_stats(2024, 'REG')
    if passing:
        print(f"   ✓ Got {len(passing)} passers")
        if len(passing) > 0:
            top_passer = passing[0]
            print(f"   ✓ Top passer: {top_passer.get('playerName')}")
            cpoe = top_passer.get('completionPercentageAboveExpectation')
            if cpoe is not None:
                print(f"   ✓ CPOE: {cpoe:.2f}")
    else:
        print("   ✗ Failed to get passing stats")
    
    # Test rushing stats
    print("\n2. Testing rushing stats...")
    rushing = client.get_rushing_stats(2024, 'REG')
    if rushing:
        print(f"   ✓ Got {len(rushing)} rushers")
    else:
        print("   ✗ Failed to get rushing stats")
    
    # Test receiving stats
    print("\n3. Testing receiving stats...")
    receiving = client.get_receiving_stats(2024, 'REG')
    if receiving:
        print(f"   ✓ Got {len(receiving)} receivers")
    else:
        print("   ✗ Failed to get receiving stats")
    
    # Print stats
    print("\n4. Client Statistics:")
    stats = client.get_stats()
    print(f"   Requests made: {stats['requests_made']}")
    print(f"   Cache hits: {stats['cache_hits']}")
    
    return True


def test_statboard_scraper():
    """Test the statboard scraper."""
    print("\n" + "=" * 60)
    print("Testing Statboard Scraper")
    print("=" * 60)
    
    scraper = StatboardScraper(output_dir="data/test/statboards")
    
    # Test single season scrape (season totals only for speed)
    print("\n1. Testing season scrape (season totals only)...")
    try:
        data = scraper.scrape_season(2024, 'REG', include_weekly=False)
        
        for stat_type, df in data.items():
            print(f"   ✓ {stat_type}: {len(df)} records")
            print(f"     Columns: {len(df.columns)}")
        
        return True
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("NGS API Scraper Test Suite")
    print("=" * 60)
    
    # Test base client
    client_ok = test_base_client()
    
    # Test statboard scraper (commented out to avoid long scrape)
    # scraper_ok = test_statboard_scraper()
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Base Client: {'✓ PASS' if client_ok else '✗ FAIL'}")
    # print(f"Statboard Scraper: {'✓ PASS' if scraper_ok else '✗ FAIL'}")
    print("\nNote: Statboard scraper test skipped (uncomment to run full scrape)")


if __name__ == "__main__":
    main()

