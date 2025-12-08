"""
Scraper for FootballDB coach records.

Coach data includes:
- Career win/loss record
- Postseason record
- Division/Conference championships
- Super Bowl wins
- Team history
"""
import logging
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd
import re

from scrapers.base_scraper import FootballDBScraper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoachScraper:
    """
    Scrape NFL head coach records from FootballDB.
    """
    
    BASE_URL = "https://www.footballdb.com/coaches"
    
    def __init__(self, output_dir: str = "data/raw/footballdb/coaches"):
        self.scraper = FootballDBScraper()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def scrape_current_coaches(self) -> List[Dict]:
        """
        Scrape records of all current NFL head coaches.
        
        Returns:
            List of coach records
        """
        url = f"{self.BASE_URL}/index.html"
        soup = self.scraper.fetch_and_parse(url)
        
        if not soup:
            logger.error("Failed to fetch coaches index")
            return []
        
        coaches = []
        
        # Find the coaches table
        table = soup.find('table')
        if not table:
            logger.error("No table found on coaches page")
            return []
        
        rows = table.find_all('tr')[1:]  # Skip header
        
        for row in rows:
            cells = row.find_all('td')
            if len(cells) < 8:
                continue
            
            # Extract coach info
            coach_link = row.find('a')
            coach_name = coach_link.get_text(strip=True) if coach_link else None
            coach_slug = coach_link.get('href', '').split('/')[-1] if coach_link else None
            
            # Parse stats
            try:
                coach = {
                    'name': coach_name,
                    'slug': coach_slug,
                    'team': cells[1].get_text(strip=True) if len(cells) > 1 else None,
                    'seasons': self._parse_int(cells[2].get_text(strip=True)) if len(cells) > 2 else None,
                    'games': self._parse_int(cells[3].get_text(strip=True)) if len(cells) > 3 else None,
                    'wins': self._parse_int(cells[4].get_text(strip=True)) if len(cells) > 4 else None,
                    'losses': self._parse_int(cells[5].get_text(strip=True)) if len(cells) > 5 else None,
                    'ties': self._parse_int(cells[6].get_text(strip=True)) if len(cells) > 6 else None,
                    'win_pct': self._parse_float(cells[7].get_text(strip=True)) if len(cells) > 7 else None,
                }
                
                # Calculate derived fields
                if coach['wins'] and coach['losses']:
                    total = coach['wins'] + coach['losses'] + (coach['ties'] or 0)
                    coach['win_rate'] = coach['wins'] / total if total > 0 else None
                
                coaches.append(coach)
                
            except Exception as e:
                logger.warning(f"Error parsing coach row: {e}")
                continue
        
        logger.info(f"Scraped {len(coaches)} current coaches")
        return coaches
    
    def scrape_coach_detail(self, coach_slug: str) -> Optional[Dict]:
        """
        Scrape detailed record for a specific coach.
        
        Args:
            coach_slug: Coach URL slug (e.g., 'bill-belichick-belicbi01')
            
        Returns:
            Dict with detailed coach record
        """
        url = f"{self.BASE_URL}/{coach_slug}"
        soup = self.scraper.fetch_and_parse(url)
        
        if not soup:
            return None
        
        coach = {
            'slug': coach_slug,
            'teams': [],
            'regular_season': {},
            'postseason': {},
            'championships': {},
        }
        
        # Parse name
        h1 = soup.find('h1')
        if h1:
            coach['name'] = h1.get_text(strip=True).replace(' Head Coaching Record', '')
        
        # Find season-by-season table
        tables = soup.find_all('table')
        
        for table in tables:
            # Check table headers to determine type
            headers = [th.get_text(strip=True) for th in table.find_all('th')]
            
            if 'Year' in headers and 'Team' in headers:
                # Season-by-season breakdown
                coach['seasons_detail'] = self._parse_seasons_table(table)
            elif 'DIV' in headers or 'CON' in headers:
                # Championship summary
                coach['championships'] = self._parse_championships(table)
        
        return coach
    
    def _parse_seasons_table(self, table) -> List[Dict]:
        """Parse season-by-season coaching record."""
        seasons = []
        
        rows = table.find_all('tr')[1:]  # Skip header
        
        for row in rows:
            cells = row.find_all('td')
            if len(cells) < 5:
                continue
            
            try:
                season = {
                    'year': self._parse_int(cells[0].get_text(strip=True)),
                    'team': cells[1].get_text(strip=True) if len(cells) > 1 else None,
                    'wins': self._parse_int(cells[2].get_text(strip=True)) if len(cells) > 2 else None,
                    'losses': self._parse_int(cells[3].get_text(strip=True)) if len(cells) > 3 else None,
                    'ties': self._parse_int(cells[4].get_text(strip=True)) if len(cells) > 4 else None,
                }
                seasons.append(season)
            except Exception as e:
                continue
        
        return seasons
    
    def _parse_championships(self, table) -> Dict:
        """Parse championship summary."""
        # Would parse division, conference, super bowl wins
        return {}
    
    def _parse_int(self, text: str) -> Optional[int]:
        """Parse integer from text."""
        try:
            return int(text.replace(',', ''))
        except (ValueError, AttributeError):
            return None
    
    def _parse_float(self, text: str) -> Optional[float]:
        """Parse float from text."""
        try:
            return float(text.replace('%', '').replace(',', ''))
        except (ValueError, AttributeError):
            return None
    
    def scrape_all_coaches(self) -> pd.DataFrame:
        """
        Scrape all current coaches and save to file.
        """
        coaches = self.scrape_current_coaches()
        
        if not coaches:
            return pd.DataFrame()
        
        df = pd.DataFrame(coaches)
        
        # Save
        output_path = self.output_dir / "current_coaches.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(df)} coaches to {output_path}")
        
        return df


if __name__ == "__main__":
    scraper = CoachScraper()
    df = scraper.scrape_all_coaches()
    print(df)

