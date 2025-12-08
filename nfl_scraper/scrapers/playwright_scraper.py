"""
Playwright-based scraper for NFL.com JavaScript-rendered pages.

Playwright is faster and more reliable than Selenium for modern web scraping.
It can also intercept network requests to discover API endpoints.
"""

import logging
import asyncio
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from scrapers.base_scraper import BaseScraper
from storage.schemas import InjuryRecord

logger = logging.getLogger(__name__)

try:
    from playwright.async_api import async_playwright, Page, Browser, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not installed. Install with: pip install playwright && playwright install")


class PlaywrightInjuryScraper(BaseScraper):
    """
    Scrape injury reports using Playwright for JavaScript-rendered content.
    
    Advantages over Selenium:
    - Faster execution
    - Better JavaScript handling
    - Can intercept network requests to discover API endpoints
    - More reliable waiting mechanisms
    """
    
    BASE_URL = "https://www.nfl.com"
    INJURIES_URL = "https://www.nfl.com/injuries/"
    
    def __init__(self, config_path: str = "config/scraping_config.yaml", headless: bool = True):
        """
        Initialize Playwright scraper.
        
        Args:
            config_path: Path to config file
            headless: Run browser in headless mode
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "Playwright not installed. Install with:\n"
                "  pip install playwright\n"
                "  playwright install chromium"
            )
        
        super().__init__(config_path)
        self.headless = headless
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.playwright = None
        
        # Store intercepted API requests
        self.intercepted_requests: List[Dict] = []
    
    async def _init_browser(self):
        """Initialize Playwright browser."""
        if self.browser:
            return
        
        self.playwright = await async_playwright().start()
        
        # Launch browser
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=['--no-sandbox', '--disable-dev-shm-usage']
        )
        
        # Create context with user agent
        user_agent = self.user_agents[0] if self.user_agents else None
        self.context = await self.browser.new_context(
            user_agent=user_agent,
            viewport={'width': 1920, 'height': 1080}
        )
        
        # Create page
        self.page = await self.context.new_page()
        
        # Intercept network requests to discover API endpoints
        self.page.on('request', self._on_request)
        self.page.on('response', self._on_response)
    
    def _on_request(self, request):
        """Intercept network requests to discover API endpoints."""
        url = request.url
        if 'api.nfl.com' in url:
            self.intercepted_requests.append({
                'type': 'request',
                'url': url,
                'method': request.method,
                'headers': request.headers,
                'timestamp': datetime.now().isoformat()
            })
            logger.debug(f"Intercepted API request: {request.method} {url}")
    
    def _on_response(self, response):
        """Intercept network responses to capture API data."""
        url = response.url
        if 'api.nfl.com' in url and response.status == 200:
            # Try to capture response body (for endpoint discovery)
            logger.debug(f"Intercepted API response: {response.status} {url}")
    
    async def _close_browser(self):
        """Close Playwright browser."""
        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        
        self.page = None
        self.context = None
        self.browser = None
        self.playwright = None
    
    async def scrape_week_async(
        self,
        season: int,
        week: int,
        wait_for_api: bool = True
    ) -> List[InjuryRecord]:
        """
        Scrape injury report for a specific week (async).
        
        Args:
            season: NFL season year
            week: Week number
            wait_for_api: Wait for API requests to complete before parsing
            
        Returns:
            List of InjuryRecord objects
        """
        from parsers.injury_parser import InjuryParser
        
        await self._init_browser()
        
        try:
            # Navigate to injuries page
            logger.info(f"Navigating to injuries page: {season} week {week}")
            await self.page.goto(self.INJURIES_URL, wait_until='networkidle')
            
            # Wait for page to load
            await self.page.wait_for_load_state('networkidle')
            
            # Try to select season and week from dropdowns
            # Look for dropdowns or buttons to change season/week
            season_selectors = [
                'select[name*="season"]',
                'select[id*="season"]',
                '[data-season]',
                'button[aria-label*="season"]'
            ]
            
            week_selectors = [
                'select[name*="week"]',
                'select[id*="week"]',
                '[data-week]',
                'button[aria-label*="week"]'
            ]
            
            # Try to set season
            for selector in season_selectors:
                try:
                    element = await self.page.query_selector(selector)
                    if element:
                        await element.select_option(str(season))
                        logger.info(f"Selected season {season}")
                        await self.page.wait_for_timeout(1000)  # Wait for update
                        break
                except Exception:
                    continue
            
            # Try to set week
            for selector in week_selectors:
                try:
                    element = await self.page.query_selector(selector)
                    if element:
                        await element.select_option(str(week))
                        logger.info(f"Selected week {week}")
                        await self.page.wait_for_timeout(2000)  # Wait for content to load
                        break
                except Exception:
                    continue
            
            # Wait for API requests if requested
            if wait_for_api:
                logger.info("Waiting for API requests to complete...")
                await self.page.wait_for_timeout(3000)
            
            # Wait for injury tables to appear
            try:
                await self.page.wait_for_selector('table', timeout=10000)
            except Exception:
                logger.warning("No tables found on page")
            
            # Get page HTML
            html = await self.page.content()
            
            # Parse HTML
            parser = InjuryParser()
            records = parser.parse_injury_page(html, season, week)
            
            logger.info(f"Parsed {len(records)} injury records")
            
            # Log intercepted API requests for discovery
            if self.intercepted_requests:
                logger.info(f"Intercepted {len(self.intercepted_requests)} API requests")
                for req in self.intercepted_requests[:5]:  # Show first 5
                    logger.debug(f"  {req['method']} {req['url']}")
            
            return records
            
        except Exception as e:
            logger.error(f"Error scraping injuries: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def scrape_week(
        self,
        season: int,
        week: int,
        wait_for_api: bool = True
    ) -> List[InjuryRecord]:
        """
        Scrape injury report (synchronous wrapper).
        
        Args:
            season: NFL season year
            week: Week number
            wait_for_api: Wait for API requests to complete
            
        Returns:
            List of InjuryRecord objects
        """
        return asyncio.run(self.scrape_week_async(season, week, wait_for_api))
    
    def scrape_season(
        self,
        season: int,
        weeks: Optional[List[int]] = None,
        wait_for_api: bool = True
    ) -> List[InjuryRecord]:
        """
        Scrape injury reports for an entire season.
        
        Args:
            season: NFL season year
            weeks: Specific weeks to scrape (default: 1-18)
            wait_for_api: Wait for API requests to complete
            
        Returns:
            List of all injury records for the season
        """
        if weeks is None:
            weeks = list(range(1, 19))
        
        all_records = []
        
        for week in weeks:
            logger.info(f"Scraping injuries: {season} week {week}")
            try:
                records = self.scrape_week(season, week, wait_for_api)
                all_records.extend(records)
                logger.info(f"  Found {len(records)} records")
            except Exception as e:
                logger.error(f"Error scraping {season} week {week}: {e}")
                continue
        
        # Close browser after scraping
        asyncio.run(self._close_browser())
        
        return all_records
    
    def discover_api_endpoints(
        self,
        season: int,
        week: int
    ) -> List[Dict]:
        """
        Discover API endpoints by intercepting network requests.
        
        This is useful for finding the actual API endpoints NFL.com uses
        for injury data.
        
        Args:
            season: NFL season year
            week: Week number
            
        Returns:
            List of intercepted API request details
        """
        logger.info(f"Discovering API endpoints for {season} week {week}")
        
        # Clear previous requests
        self.intercepted_requests = []
        
        # Scrape and capture requests
        self.scrape_week(season, week, wait_for_api=True)
        
        # Filter for relevant endpoints
        injury_endpoints = [
            req for req in self.intercepted_requests
            if 'injur' in req['url'].lower() or 'game' in req['url'].lower()
        ]
        
        return injury_endpoints
    
    def __del__(self):
        """Cleanup on deletion."""
        if self.browser:
            try:
                asyncio.run(self._close_browser())
            except:
                pass

