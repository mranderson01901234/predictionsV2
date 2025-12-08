"""
Scraper for NFL.com injury reports.
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers.base_scraper import BaseScraper
from parsers.injury_parser import InjuryParser
from storage.schemas import InjuryRecord

logger = logging.getLogger(__name__)


class InjuryScraper(BaseScraper):
    """
    Scrape injury reports from NFL.com.
    
    NFL.com injury page uses JavaScript to load data for different weeks/seasons.
    We may need to:
    1. Use Selenium/Playwright for dynamic content, OR
    2. Find the underlying API endpoint, OR  
    3. Construct direct URLs if they exist
    
    For historical data (2015-2024), we'll need to find archived pages or API.
    """
    
    BASE_URL = "https://www.nfl.com"
    INJURIES_URL = "https://www.nfl.com/injuries/"
    
    def __init__(self, config_path: str = "config/scraping_config.yaml"):
        super().__init__(config_path)
        self.parser = InjuryParser()
    
    def scrape_week(
        self, 
        season: int, 
        week: int,
        use_cache: bool = True
    ) -> List[InjuryRecord]:
        """
        Scrape injury report for a specific week.
        
        Note: NFL.com may require JavaScript rendering or API calls.
        This implementation assumes static HTML or will need enhancement.
        """
        # Try direct URL first (may not work for historical data)
        # NFL.com uses dropdowns to select season/week
        
        # Option 1: Try direct URL pattern
        url = f"{self.INJURIES_URL}?season={season}&week={week}"
        
        html = self.fetch(url, use_cache=use_cache)
        
        if not html:
            logger.warning(f"Failed to fetch injuries for {season} week {week}")
            return []
        
        return self.parser.parse_injury_page(html, season, week)
    
    def scrape_season(
        self,
        season: int,
        weeks: Optional[List[int]] = None,
        use_cache: bool = True
    ) -> List[InjuryRecord]:
        """
        Scrape injury reports for an entire season.
        
        Args:
            season: NFL season year
            weeks: Specific weeks to scrape (default: 1-18)
            use_cache: Whether to use cached responses
            
        Returns:
            List of all injury records for the season
        """
        if weeks is None:
            # Regular season weeks
            weeks = list(range(1, 19))
        
        all_records = []
        
        for week in weeks:
            logger.info(f"Scraping injuries: {season} week {week}")
            
            try:
                records = self.scrape_week(season, week, use_cache)
                all_records.extend(records)
                logger.info(f"  Found {len(records)} records")
            except Exception as e:
                logger.error(f"Error scraping {season} week {week}: {e}")
                continue
        
        return all_records
    
    def scrape_historical(
        self,
        start_season: int = 2015,
        end_season: int = 2024,
        use_cache: bool = True
    ) -> List[InjuryRecord]:
        """
        Scrape historical injury data for multiple seasons.
        
        This is a long-running operation. Progress is logged.
        """
        all_records = []
        
        for season in range(start_season, end_season + 1):
            logger.info(f"=== Scraping season {season} ===")
            
            season_records = self.scrape_season(season, use_cache=use_cache)
            all_records.extend(season_records)
            
            logger.info(f"Season {season}: {len(season_records)} total records")
        
        logger.info(f"Historical scrape complete: {len(all_records)} total records")
        return all_records


class SeleniumInjuryScraper(InjuryScraper):
    """
    Enhanced injury scraper using Selenium for JavaScript-rendered content.
    
    Use this if the basic scraper fails due to dynamic content loading.
    """
    
    def __init__(self, config_path: str = "config/scraping_config.yaml"):
        super().__init__(config_path)
        self.driver = None
    
    def _init_driver(self):
        """Initialize Selenium WebDriver."""
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument(f'user-agent={self.user_agents[0]}')
        
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )
    
    def _close_driver(self):
        """Close Selenium WebDriver."""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def scrape_week_selenium(
        self,
        season: int,
        week: int
    ) -> List[InjuryRecord]:
        """
        Scrape injury report using Selenium for dynamic content.
        """
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait, Select
        from selenium.webdriver.support import expected_conditions as EC
        import time
        
        if not self.driver:
            self._init_driver()
        
        try:
            # Navigate to injuries page
            self.driver.get(self.INJURIES_URL)
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
            
            # Select season from dropdown
            season_dropdown = Select(self.driver.find_element(By.ID, "season-select"))
            season_dropdown.select_by_value(str(season))
            
            # Wait for update
            time.sleep(1)
            
            # Select week from dropdown
            week_dropdown = Select(self.driver.find_element(By.ID, "week-select"))
            week_dropdown.select_by_value(str(week))
            
            # Wait for content to load
            time.sleep(2)
            
            # Get page source
            html = self.driver.page_source
            
            return self.parser.parse_injury_page(html, season, week)
            
        except Exception as e:
            logger.error(f"Selenium scrape error: {e}")
            return []

