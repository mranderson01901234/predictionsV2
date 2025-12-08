"""
Scraper for NFL.com player stats pages.

Targets:
- Career stats: /players/{slug}/stats/career
- Situational stats: /players/{slug}/stats/situational/{year}/
- Game logs: /players/{slug}/stats/logs/{year}/
- Splits: /players/{slug}/stats/splits/{year}/
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers.base_scraper import BaseScraper
from storage.schemas import PlayerCareerStats, PlayerSituationalStats

logger = logging.getLogger(__name__)


class PlayerStatsScraper(BaseScraper):
    """
    Scrape player statistics from NFL.com.
    """
    
    BASE_URL = "https://www.nfl.com"
    
    def __init__(self, config_path: str = "config/scraping_config.yaml"):
        super().__init__(config_path)
    
    def get_player_url(self, player_slug: str, stat_type: str, season: Optional[int] = None) -> str:
        """
        Construct URL for player stats page.
        
        Args:
            player_slug: Player URL slug (e.g., "patrick-mahomes")
            stat_type: Type of stats ("career", "situational", "logs", "splits")
            season: Season year (required for situational, logs, splits)
        """
        base = f"{self.BASE_URL}/players/{player_slug}/stats"
        
        if stat_type == "career":
            return f"{base}/career"
        elif stat_type == "summary":
            return base
        elif stat_type in ["situational", "logs", "splits"]:
            if season is None:
                raise ValueError(f"Season required for {stat_type} stats")
            return f"{base}/{stat_type}/{season}/"
        else:
            raise ValueError(f"Unknown stat type: {stat_type}")
    
    def scrape_career_stats(self, player_slug: str) -> Optional[Dict]:
        """
        Scrape career stats for a player.
        
        Returns dict with career totals and season-by-season breakdown.
        """
        url = self.get_player_url(player_slug, "career")
        html = self.fetch(url)
        
        if not html:
            return None
        
        return self._parse_career_stats(html, player_slug)
    
    def scrape_situational_stats(
        self, 
        player_slug: str, 
        season: int
    ) -> Optional[Dict]:
        """
        Scrape situational stats for a player and season.
        
        Returns dict with stats by:
        - Quarter (1st, 2nd, 3rd, 4th, 4th within 7)
        - Point differential (ahead, behind, tied)
        - Home/away
        - Half (1st, 2nd)
        - Field position
        - Stadium surface
        """
        url = self.get_player_url(player_slug, "situational", season)
        html = self.fetch(url)
        
        if not html:
            return None
        
        return self._parse_situational_stats(html, player_slug, season)
    
    def scrape_game_logs(
        self,
        player_slug: str,
        season: int
    ) -> Optional[List[Dict]]:
        """
        Scrape game-by-game stats for a player and season.
        """
        url = self.get_player_url(player_slug, "logs", season)
        html = self.fetch(url)
        
        if not html:
            return None
        
        return self._parse_game_logs(html, player_slug, season)
    
    def _parse_career_stats(self, html: str, player_slug: str) -> Dict:
        """Parse career stats HTML."""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html, 'html.parser')
        
        stats = {
            'player_slug': player_slug,
            'seasons': [],
            'career_totals': {},
        }
        
        # Find career stats table
        tables = soup.find_all('table')
        
        for table in tables:
            # Look for the main career stats table
            headers = [th.get_text(strip=True) for th in table.find_all('th')]
            
            if 'SEASON' in headers or 'Season' in headers:
                rows = table.find_all('tr')[1:]  # Skip header
                
                for row in rows:
                    cells = row.find_all('td')
                    if not cells:
                        continue
                    
                    # Check if this is a totals row
                    first_cell = cells[0].get_text(strip=True)
                    
                    if first_cell.upper() == 'TOTAL':
                        # Parse totals row
                        stats['career_totals'] = self._parse_stat_row(headers, cells)
                    else:
                        # Parse season row
                        season_stats = self._parse_stat_row(headers, cells)
                        if season_stats:
                            stats['seasons'].append(season_stats)
        
        return stats
    
    def _parse_situational_stats(self, html: str, player_slug: str, season: int) -> Dict:
        """Parse situational stats HTML."""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html, 'html.parser')
        
        stats = {
            'player_slug': player_slug,
            'season': season,
            'by_quarter': {},
            'by_point_differential': {},
            'by_home_away': {},
            'by_half': {},
            'by_field_position': {},
        }
        
        # The page has multiple tables for different situational breakdowns
        # Each table has a header like "Quarters", "Point Differential", etc.
        
        tables = soup.find_all('table')
        current_section = None
        
        for table in tables:
            # Try to find section header
            prev_sibling = table.find_previous_sibling(['h3', 'h4', 'div'])
            if prev_sibling:
                section_text = prev_sibling.get_text(strip=True).lower()
                if 'quarter' in section_text:
                    current_section = 'by_quarter'
                elif 'point' in section_text or 'differential' in section_text:
                    current_section = 'by_point_differential'
                elif 'home' in section_text or 'road' in section_text:
                    current_section = 'by_home_away'
                elif 'half' in section_text:
                    current_section = 'by_half'
                elif 'field' in section_text or 'position' in section_text:
                    current_section = 'by_field_position'
            
            if current_section:
                headers = [th.get_text(strip=True) for th in table.find_all('th')]
                rows = table.find_all('tr')[1:]
                
                for row in rows:
                    cells = row.find_all('td')
                    if cells:
                        row_label = cells[0].get_text(strip=True)
                        row_stats = self._parse_stat_row(headers, cells)
                        if row_stats:
                            stats[current_section][row_label] = row_stats
        
        return stats
    
    def _parse_game_logs(self, html: str, player_slug: str, season: int) -> List[Dict]:
        """Parse game logs HTML."""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html, 'html.parser')
        
        logs = []
        
        # Find game log table
        tables = soup.find_all('table')
        
        for table in tables:
            headers = [th.get_text(strip=True) for th in table.find_all('th')]
            
            if 'WK' in headers or 'OPP' in headers:
                rows = table.find_all('tr')[1:]
                
                for row in rows:
                    cells = row.find_all('td')
                    if cells:
                        game_stats = self._parse_stat_row(headers, cells)
                        if game_stats:
                            game_stats['player_slug'] = player_slug
                            game_stats['season'] = season
                            logs.append(game_stats)
        
        return logs
    
    def _parse_stat_row(self, headers: List[str], cells) -> Optional[Dict]:
        """
        Parse a single row of stats into a dict.
        """
        if len(cells) != len(headers):
            # Header/cell count mismatch
            return None
        
        row_data = {}
        
        for header, cell in zip(headers, cells):
            # Normalize header
            key = header.lower().replace(' ', '_').replace('%', 'pct')
            key = re.sub(r'[^a-z0-9_]', '', key)
            
            # Get value
            value = cell.get_text(strip=True)
            
            # Try to convert to number
            try:
                if '.' in value:
                    value = float(value)
                elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                    value = int(value)
            except (ValueError, AttributeError):
                pass  # Keep as string
            
            row_data[key] = value
        
        return row_data
    
    def scrape_all_qb_stats(
        self,
        qb_slugs: List[str],
        seasons: List[int]
    ) -> Dict[str, Dict]:
        """
        Scrape stats for multiple QBs across multiple seasons.
        
        Args:
            qb_slugs: List of QB player slugs
            seasons: List of seasons to scrape
            
        Returns:
            Dict mapping player_slug to their stats
        """
        all_stats = {}
        
        for slug in qb_slugs:
            logger.info(f"Scraping stats for {slug}")
            
            player_stats = {
                'career': self.scrape_career_stats(slug),
                'situational': {},
                'game_logs': {},
            }
            
            for season in seasons:
                player_stats['situational'][season] = self.scrape_situational_stats(slug, season)
                player_stats['game_logs'][season] = self.scrape_game_logs(slug, season)
            
            all_stats[slug] = player_stats
        
        return all_stats

