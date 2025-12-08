"""
Parser for FootballDB statistics tables.

FootballDB uses consistent HTML table structure across pages.
Tables have:
- Header row with column names
- Data rows with player/team stats
- Sortable columns with data attributes
"""
import re
import logging
from typing import List, Dict, Optional, Any
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatsTableParser:
    """
    Parse FootballDB statistics tables.
    
    Handles:
    - Player split tables
    - Team split tables
    - Various stat categories (passing, rushing, receiving, etc.)
    """
    
    def parse_stats_table(
        self, 
        soup: BeautifulSoup,
        table_class: str = "statistics"
    ) -> List[Dict[str, Any]]:
        """
        Parse a statistics table from the page.
        
        Args:
            soup: BeautifulSoup object of the page
            table_class: CSS class of the stats table
            
        Returns:
            List of dicts, one per row
        """
        # Find the stats table
        table = soup.find('table', class_=table_class)
        
        if not table:
            # Try finding any table with statistics
            tables = soup.find_all('table')
            for t in tables:
                if t.find('thead') and t.find('tbody'):
                    table = t
                    break
        
        if not table:
            logger.warning("No statistics table found")
            return []
        
        # Get headers
        headers = self._extract_headers(table)
        if not headers:
            logger.warning("No headers found in table")
            return []
        
        # Parse rows
        records = []
        tbody = table.find('tbody')
        
        if tbody:
            rows = tbody.find_all('tr')
        else:
            rows = table.find_all('tr')[1:]  # Skip header row
        
        for row in rows:
            record = self._parse_row(row, headers)
            if record:
                records.append(record)
        
        logger.info(f"Parsed {len(records)} records from table")
        return records
    
    def _extract_headers(self, table: BeautifulSoup) -> List[str]:
        """Extract column headers from table."""
        headers = []
        
        thead = table.find('thead')
        if thead:
            header_row = thead.find('tr')
        else:
            header_row = table.find('tr')
        
        if not header_row:
            return []
        
        for th in header_row.find_all(['th', 'td']):
            # Get header text
            text = th.get_text(strip=True)
            
            # Skip empty headers
            if not text:
                headers.append(f'col_{len(headers)}')  # Use generic name
                continue
            
            # Clean up header name
            header = self._clean_header(text)
            
            # If still empty after cleaning, use generic name
            if not header:
                header = f'col_{len(headers)}'
            
            headers.append(header)
        
        return headers
    
    def _clean_header(self, text: str) -> str:
        """Clean header text to create valid column name."""
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Convert to snake_case
        text = text.lower().strip()
        text = re.sub(r'\s+', '_', text)
        
        # Handle common abbreviations
        header_map = {
            'player': 'player_name',
            'team': 'team',
            'gm': 'games',
            'att': 'attempts',
            'comp': 'completions',
            'pct': 'percentage',
            'yds': 'yards',
            'td': 'touchdowns',
            'int': 'interceptions',
            'sck': 'sacks',
            'rate': 'rating',
            'avg': 'average',
            'lng': 'longest',
            'rec': 'receptions',
            'tgt': 'targets',
            'car': 'carries',
            'fgm': 'field_goals_made',
            'fga': 'field_goals_attempted',
            'xpm': 'extra_points_made',
            'xpa': 'extra_points_attempted',
            'pts': 'points',
        }
        
        # Check if we need to expand abbreviation
        for abbr, full in header_map.items():
            if text == abbr:
                return full
        
        return text
    
    def _parse_row(
        self, 
        row: BeautifulSoup, 
        headers: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Parse a single data row."""
        cells = row.find_all(['td', 'th'])
        
        if len(cells) == 0:
            return None
        
        # Skip header rows (rows that are all th elements)
        if all(cell.name == 'th' for cell in cells):
            return None
        
        record = {}
        
        # Only process up to the number of headers we have
        num_cells = min(len(cells), len(headers))
        
        for i in range(num_cells):
            header = headers[i]
            cell = cells[i]
            value = self._parse_cell(cell, header)
            
            # Only add non-None values, or if it's a key field like player_name
            if value is not None or header in ['player_name', 'team']:
                record[header] = value
        
        # Extract player/team link if present
        link = row.find('a')
        if link and link.get('href'):
            href = link.get('href')
            
            if '/players/' in href:
                record['player_slug'] = href.split('/players/')[-1].rstrip('/')
            elif '/teams/' in href:
                record['team_slug'] = href.split('/teams/')[-1].rstrip('/')
        
        # Only return if we have at least some data
        if not record or len(record) < 2:
            return None
        
        return record
    
    def _parse_cell(self, cell: BeautifulSoup, header: str) -> Any:
        """Parse cell value with type inference."""
        text = cell.get_text(strip=True)
        
        # Clean up text - remove non-breaking spaces and other whitespace issues
        text = text.replace('\xa0', ' ').strip()
        
        if not text or text == '-' or text == '--':
            return None
        
        # For player_name and team columns, always keep as string
        if header in ['player_name', 'team', 'player_slug', 'team_slug']:
            return text
        
        # Try to parse as number
        try:
            # Remove commas from numbers
            text_clean = text.replace(',', '').replace(' ', '')
            
            # Check for percentage
            if '%' in text_clean:
                return float(text_clean.replace('%', ''))
            
            # Check for decimal
            if '.' in text_clean:
                return float(text_clean)
            
            # Integer
            return int(text_clean)
            
        except (ValueError, AttributeError):
            # Keep as string if parsing fails
            return text
    
    def parse_player_splits_page(
        self, 
        soup: BeautifulSoup
    ) -> Dict[str, Any]:
        """
        Parse a player splits page.
        
        Returns dict with:
        - split_type: Type of split (home/away, etc.)
        - stat_type: Stat category (passing, rushing, etc.)
        - year: Season year
        - records: List of player records
        """
        result = {
            'split_type': None,
            'stat_type': None,
            'year': None,
            'records': []
        }
        
        # Extract page info from title
        title = soup.find('title')
        if title:
            title_text = title.get_text()
            # Parse: "2025 NFL Passing Splits - Home Games"
            year_match = re.search(r'(\d{4})', title_text)
            if year_match:
                result['year'] = int(year_match.group(1))
        
        # Parse the stats table
        result['records'] = self.parse_stats_table(soup)
        
        return result
    
    def parse_team_splits_page(
        self, 
        soup: BeautifulSoup
    ) -> Dict[str, Any]:
        """Parse a team splits page."""
        # Same structure as player splits
        return self.parse_player_splits_page(soup)


# Column reference for each stat type
PASSING_COLUMNS = [
    'player_name', 'team', 'games', 'completions', 'attempts', 
    'percentage', 'yards', 'yards_per_attempt', 'touchdowns',
    'interceptions', 'sacks', 'sack_yards', 'rating'
]

RUSHING_COLUMNS = [
    'player_name', 'team', 'games', 'carries', 'yards',
    'average', 'longest', 'touchdowns', 'first_downs',
    'fumbles', 'fumbles_lost'
]

RECEIVING_COLUMNS = [
    'player_name', 'team', 'games', 'receptions', 'targets',
    'yards', 'average', 'longest', 'touchdowns', 'first_downs'
]

KICKING_COLUMNS = [
    'player_name', 'team', 'games', 'field_goals_made', 'field_goals_attempted',
    'fg_percentage', 'longest', 'extra_points_made', 'extra_points_attempted',
    'xp_percentage', 'points'
]

DEFENSE_COLUMNS = [
    'team', 'games', 'points_allowed', 'yards_allowed',
    'passing_yards_allowed', 'rushing_yards_allowed',
    'takeaways', 'sacks', 'interceptions'
]

