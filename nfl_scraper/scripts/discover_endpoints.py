"""
Script to discover NFL.com API endpoints using Playwright network interception.
"""

import sys
import logging
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers.playwright_scraper import PlaywrightInjuryScraper

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Discover API endpoints."""
    logger.info("=" * 60)
    logger.info("NFL.com API Endpoint Discovery")
    logger.info("=" * 60)
    
    try:
        scraper = PlaywrightInjuryScraper(headless=False)  # Show browser for debugging
        
        # Discover endpoints for current week
        logger.info("\nDiscovering endpoints for 2024 week 1...")
        endpoints = scraper.discover_api_endpoints(2024, 1)
        
        if endpoints:
            logger.info(f"\n✓ Found {len(endpoints)} relevant API endpoints:")
            for i, endpoint in enumerate(endpoints, 1):
                logger.info(f"\n{i}. {endpoint['method']} {endpoint['url']}")
                logger.info(f"   Headers: {json.dumps(endpoint.get('headers', {}), indent=6)[:200]}")
        else:
            logger.warning("No endpoints discovered. Check browser console.")
        
        # Save discovered endpoints
        output_file = Path(__file__).parent.parent / "data" / "raw" / "discovered_endpoints.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump({
                'all_requests': scraper.intercepted_requests,
                'injury_endpoints': endpoints
            }, f, indent=2)
        
        logger.info(f"\n✓ Saved discovered endpoints to {output_file}")
        
    except ImportError as e:
        logger.error(f"Playwright not installed: {e}")
        logger.info("\nInstall with:")
        logger.info("  pip install playwright")
        logger.info("  playwright install chromium")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

