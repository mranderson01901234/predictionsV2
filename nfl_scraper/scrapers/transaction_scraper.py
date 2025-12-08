"""
Scraper for NFL.com transactions.

Types:
- Trades
- Signings
- Reserve List (IR, PUP, etc.)
- Waivers
- Terminations
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers.base_scraper import BaseScraper
from storage.schemas import TransactionRecord

logger = logging.getLogger(__name__)


class TransactionScraper(BaseScraper):
    """
    Scrape transaction data from NFL.com.
    """
    
    BASE_URL = "https://www.nfl.com"
    
    TRANSACTION_TYPES = ['trades', 'signings', 'reserve-list', 'waivers', 'terminations']
    MONTHS = ['january', 'february', 'march', 'april', 'may', 'june',
              'july', 'august', 'september', 'october', 'november', 'december']
    
    def __init__(self, config_path: str = "config/scraping_config.yaml"):
        super().__init__(config_path)
    
    def get_transaction_url(
        self,
        trans_type: str,
        season: int,
        month: int
    ) -> str:
        """Construct URL for transaction page."""
        month_num = str(month).zfill(2)
        return f"{self.BASE_URL}/transactions/league/{trans_type}/{season}/{month_num}"
    
    def scrape_transactions(
        self,
        trans_type: str,
        season: int,
        month: int
    ) -> List[TransactionRecord]:
        """
        Scrape transactions for a specific type, season, and month.
        """
        url = self.get_transaction_url(trans_type, season, month)
        html = self.fetch(url)
        
        if not html:
            return []
        
        return self._parse_transactions(html, trans_type, season, month)
    
    def scrape_season_transactions(
        self,
        season: int,
        trans_types: Optional[List[str]] = None
    ) -> List[TransactionRecord]:
        """
        Scrape all transactions for a season.
        """
        if trans_types is None:
            trans_types = self.TRANSACTION_TYPES
        
        all_transactions = []
        
        for trans_type in trans_types:
            for month in range(1, 13):
                logger.info(f"Scraping {trans_type} for {season}/{month:02d}")
                
                transactions = self.scrape_transactions(trans_type, season, month)
                all_transactions.extend(transactions)
        
        return all_transactions
    
    def _parse_transactions(
        self,
        html: str,
        trans_type: str,
        season: int,
        month: int
    ) -> List[TransactionRecord]:
        """Parse transaction page HTML."""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html, 'html.parser')
        records = []
        
        # Find transaction table
        tables = soup.find_all('table')
        
        for table in tables:
            headers = [th.get_text(strip=True).lower() for th in table.find_all('th')]
            
            # Check if this looks like a transaction table
            if 'name' in headers or 'player' in headers:
                rows = table.find_all('tr')[1:]
                
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) < 3:
                        continue
                    
                    try:
                        record = self._parse_transaction_row(
                            cells, headers, trans_type, season, month
                        )
                        if record:
                            records.append(record)
                    except Exception as e:
                        logger.warning(f"Error parsing transaction row: {e}")
                        continue
        
        return records
    
    def _parse_transaction_row(
        self,
        cells,
        headers: List[str],
        trans_type: str,
        season: int,
        month: int
    ) -> Optional[TransactionRecord]:
        """Parse a single transaction row."""
        from datetime import date
        
        # Extract data based on columns
        data = {}
        
        for i, (header, cell) in enumerate(zip(headers, cells)):
            text = cell.get_text(strip=True)
            link = cell.find('a')
            
            if 'from' in header:
                data['from_team'] = self._extract_team(cell)
            elif 'to' in header:
                data['to_team'] = self._extract_team(cell)
            elif 'date' in header:
                data['date'] = self._parse_date(text, season, month)
            elif 'name' in header or 'player' in header:
                data['player_name'] = text
                if link:
                    href = link.get('href', '')
                    match = re.search(r'/players/([^/]+)/', href)
                    data['player_id'] = match.group(1) if match else ''
            elif 'position' in header:
                data['position'] = text
            elif 'transaction' in header:
                data['details'] = text
        
        if 'player_name' not in data:
            return None
        
        return TransactionRecord(
            date=data.get('date', date(season, month, 1)),
            transaction_type=trans_type,
            from_team=data.get('from_team'),
            to_team=data.get('to_team'),
            player_name=data['player_name'],
            player_id=data.get('player_id', ''),
            position=data.get('position'),
            details=data.get('details'),
        )
    
    def _extract_team(self, cell) -> Optional[str]:
        """Extract team abbreviation from cell."""
        link = cell.find('a')
        if link:
            href = link.get('href', '')
            match = re.search(r'/teams/([^/]+)/', href)
            if match:
                return self._slug_to_abbr(match.group(1))
        return cell.get_text(strip=True) or None
    
    def _parse_date(self, text: str, season: int, month: int) -> 'date':
        """Parse date from text like '11/04'."""
        from datetime import date
        
        match = re.search(r'(\d{1,2})/(\d{1,2})', text)
        if match:
            m, d = int(match.group(1)), int(match.group(2))
            # Handle year rollover (transactions in Jan-Feb might be for prev season)
            year = season if m >= 3 else season + 1
            return date(year, m, d)
        
        return date(season, month, 1)
    
    def _slug_to_abbr(self, slug: str) -> str:
        """Convert team slug to abbreviation."""
        TEAM_MAP = {
            'arizona-cardinals': 'ARI', 'atlanta-falcons': 'ATL',
            'baltimore-ravens': 'BAL', 'buffalo-bills': 'BUF',
            'carolina-panthers': 'CAR', 'chicago-bears': 'CHI',
            'cincinnati-bengals': 'CIN', 'cleveland-browns': 'CLE',
            'dallas-cowboys': 'DAL', 'denver-broncos': 'DEN',
            'detroit-lions': 'DET', 'green-bay-packers': 'GB',
            'houston-texans': 'HOU', 'indianapolis-colts': 'IND',
            'jacksonville-jaguars': 'JAX', 'kansas-city-chiefs': 'KC',
            'las-vegas-raiders': 'LV', 'los-angeles-chargers': 'LAC',
            'los-angeles-rams': 'LAR', 'miami-dolphins': 'MIA',
            'minnesota-vikings': 'MIN', 'new-england-patriots': 'NE',
            'new-orleans-saints': 'NO', 'new-york-giants': 'NYG',
            'new-york-jets': 'NYJ', 'philadelphia-eagles': 'PHI',
            'pittsburgh-steelers': 'PIT', 'san-francisco-49ers': 'SF',
            'seattle-seahawks': 'SEA', 'tampa-bay-buccaneers': 'TB',
            'tennessee-titans': 'TEN', 'washington-commanders': 'WAS',
        }
        return TEAM_MAP.get(slug, slug.upper()[:3])

