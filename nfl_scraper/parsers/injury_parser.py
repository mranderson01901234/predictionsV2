"""
Parser for NFL.com injury report pages.
"""

import re
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import date
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.schemas import InjuryRecord

logger = logging.getLogger(__name__)


class InjuryParser:
    """
    Parse NFL.com injury report HTML.
    
    The injury page contains tables for each game, with columns:
    - Player (with link to player page)
    - Position
    - Injuries (body part: "Hamstring", "Knee, Ankle", etc.)
    - Practice Status ("Full Participation in Practice", "Limited...", "Did Not Participate...")
    - Game Status ("Out", "Doubtful", "Questionable", or empty)
    """
    
    # Practice status mapping
    PRACTICE_STATUS_MAP = {
        'full participation in practice': 'Full',
        'full': 'Full',
        'limited participation in practice': 'Limited',
        'limited': 'Limited',
        'did not participate in practice': 'DNP',
        'did not participate': 'DNP',
        'dnp': 'DNP',
    }
    
    # Game status normalization
    GAME_STATUS_MAP = {
        'out': 'Out',
        'doubtful': 'Doubtful',
        'questionable': 'Questionable',
        'probable': 'Probable',
    }
    
    def __init__(self):
        pass
    
    def parse_injury_page(
        self, 
        html: str, 
        season: int, 
        week: int
    ) -> List[InjuryRecord]:
        """
        Parse full injury report page.
        
        Args:
            html: Raw HTML content
            season: NFL season year
            week: Week number
            
        Returns:
            List of InjuryRecord objects
        """
        soup = BeautifulSoup(html, 'html.parser')
        records = []
        
        # Find all game sections
        # Each game has a header with teams and a table with injuries
        game_sections = self._find_game_sections(soup)
        
        for game_info, injury_table in game_sections:
            game_records = self._parse_game_injuries(
                game_info, injury_table, season, week
            )
            records.extend(game_records)
        
        logger.info(f"Parsed {len(records)} injury records for {season} week {week}")
        return records
    
    def _find_game_sections(self, soup: BeautifulSoup) -> List[Tuple[Dict, BeautifulSoup]]:
        """
        Find all game sections on the page.
        
        Returns list of (game_info, injury_table) tuples.
        """
        sections = []
        
        # Look for game containers - structure may vary
        # The page has sections like "THURSDAY, DECEMBER 4TH" with games underneath
        
        # Find all tables with injury data
        tables = soup.find_all('table')
        
        for table in tables:
            # Check if this is an injury table (has expected columns)
            headers = table.find_all('th')
            header_text = [h.get_text(strip=True).lower() for h in headers]
            
            if 'player' in header_text and 'injuries' in header_text:
                # Find the associated team info
                # Usually in a preceding element
                game_info = self._extract_game_info(table)
                if game_info:
                    sections.append((game_info, table))
        
        return sections
    
    def _extract_game_info(self, table: BeautifulSoup) -> Optional[Dict]:
        """
        Extract game information (teams, date) from context around table.
        """
        # Look for team names in preceding elements
        # The structure has team abbreviations like "DAL Cowboys" and "DET Lions"
        
        # Find parent container
        parent = table.find_parent()
        
        # Look for team links or text
        team_info = {
            'home_team': None,
            'away_team': None,
            'game_date': None,
        }
        
        # This will need to be refined based on actual HTML structure
        # For now, return basic structure
        
        # Try to find team abbreviations
        team_links = parent.find_all('a', href=re.compile(r'/teams/'))
        if len(team_links) >= 2:
            # Extract team from URL like /teams/dallas-cowboys/
            for link in team_links:
                href = link.get('href', '')
                team_match = re.search(r'/teams/([^/]+)/', href)
                if team_match:
                    team_slug = team_match.group(1)
                    team_abbr = self._slug_to_abbr(team_slug)
                    if not team_info['away_team']:
                        team_info['away_team'] = team_abbr
                    else:
                        team_info['home_team'] = team_abbr
        
        return team_info if team_info['home_team'] else None
    
    def _parse_game_injuries(
        self,
        game_info: Dict,
        table: BeautifulSoup,
        season: int,
        week: int
    ) -> List[InjuryRecord]:
        """
        Parse injuries for a single game.
        """
        records = []
        
        # Find all rows (skip header)
        rows = table.find_all('tr')[1:]  # Skip header row
        
        for row in rows:
            cells = row.find_all('td')
            if len(cells) < 5:
                continue
            
            try:
                # Extract player info
                player_cell = cells[0]
                player_link = player_cell.find('a')
                
                if player_link:
                    player_name = player_link.get_text(strip=True)
                    player_href = player_link.get('href', '')
                    player_id = self._extract_player_id(player_href)
                else:
                    player_name = player_cell.get_text(strip=True)
                    player_id = self._name_to_id(player_name)
                
                # Position
                position = cells[1].get_text(strip=True)
                
                # Injuries (can be multiple: "Toe, Ankle")
                injuries_text = cells[2].get_text(strip=True)
                injury_types, is_resting = self._parse_injury_types(injuries_text)
                
                # Practice status
                practice_text = cells[3].get_text(strip=True)
                practice_status = self._normalize_practice_status(practice_text)
                
                # Game status
                game_status_text = cells[4].get_text(strip=True)
                game_status = self._normalize_game_status(game_status_text)
                
                # Determine which team this player is on
                # (usually indicated by table header or section)
                team = game_info.get('home_team') or game_info.get('away_team')
                opponent = game_info.get('away_team') if team == game_info.get('home_team') else game_info.get('home_team')
                
                record = InjuryRecord(
                    season=season,
                    week=week,
                    game_date=None,  # Would need to parse from page
                    team=team,
                    opponent=opponent,
                    player_name=player_name,
                    player_id=player_id,
                    position=position,
                    injury_type=injury_types[0] if injury_types else None,
                    injury_types=injury_types,
                    practice_status_wed=None,  # Would need historical data
                    practice_status_thu=None,
                    practice_status_fri=None,
                    practice_status_final=practice_status,
                    game_status=game_status,
                    is_resting=is_resting,
                )
                
                records.append(record)
                
            except Exception as e:
                logger.warning(f"Error parsing injury row: {e}")
                continue
        
        return records
    
    def _parse_injury_types(self, text: str) -> Tuple[List[str], bool]:
        """
        Parse injury types from text like "Knee", "Toe, Ankle", 
        "Shoulder, Not injury related - resting player".
        
        Returns (injury_types, is_resting)
        """
        is_resting = 'not injury related' in text.lower() or 'resting' in text.lower()
        
        # Remove resting notes
        clean_text = re.sub(r'not injury related.*', '', text, flags=re.IGNORECASE)
        clean_text = re.sub(r'resting.*', '', clean_text, flags=re.IGNORECASE)
        
        # Split by comma
        injuries = [i.strip() for i in clean_text.split(',') if i.strip()]
        
        return injuries, is_resting
    
    def _normalize_practice_status(self, text: str) -> str:
        """Normalize practice status to DNP/Limited/Full."""
        text_lower = text.lower().strip()
        
        for pattern, normalized in self.PRACTICE_STATUS_MAP.items():
            if pattern in text_lower:
                return normalized
        
        return text  # Return original if no match
    
    def _normalize_game_status(self, text: str) -> Optional[str]:
        """Normalize game status to Out/Doubtful/Questionable/Probable."""
        if not text:
            return None
        
        text_lower = text.lower().strip()
        
        for pattern, normalized in self.GAME_STATUS_MAP.items():
            if pattern in text_lower:
                return normalized
        
        return text if text else None
    
    def _extract_player_id(self, href: str) -> str:
        """Extract player ID from URL like /players/patrick-mahomes/."""
        match = re.search(r'/players/([^/]+)/', href)
        return match.group(1) if match else ''
    
    def _name_to_id(self, name: str) -> str:
        """Convert player name to URL-friendly ID."""
        return name.lower().replace(' ', '-').replace('.', '').replace("'", '')
    
    def _slug_to_abbr(self, slug: str) -> str:
        """Convert team slug to abbreviation."""
        TEAM_MAP = {
            'arizona-cardinals': 'ARI',
            'atlanta-falcons': 'ATL',
            'baltimore-ravens': 'BAL',
            'buffalo-bills': 'BUF',
            'carolina-panthers': 'CAR',
            'chicago-bears': 'CHI',
            'cincinnati-bengals': 'CIN',
            'cleveland-browns': 'CLE',
            'dallas-cowboys': 'DAL',
            'denver-broncos': 'DEN',
            'detroit-lions': 'DET',
            'green-bay-packers': 'GB',
            'houston-texans': 'HOU',
            'indianapolis-colts': 'IND',
            'jacksonville-jaguars': 'JAX',
            'kansas-city-chiefs': 'KC',
            'las-vegas-raiders': 'LV',
            'los-angeles-chargers': 'LAC',
            'los-angeles-rams': 'LAR',
            'miami-dolphins': 'MIA',
            'minnesota-vikings': 'MIN',
            'new-england-patriots': 'NE',
            'new-orleans-saints': 'NO',
            'new-york-giants': 'NYG',
            'new-york-jets': 'NYJ',
            'philadelphia-eagles': 'PHI',
            'pittsburgh-steelers': 'PIT',
            'san-francisco-49ers': 'SF',
            'seattle-seahawks': 'SEA',
            'tampa-bay-buccaneers': 'TB',
            'tennessee-titans': 'TEN',
            'washington-commanders': 'WAS',
        }
        return TEAM_MAP.get(slug, slug.upper()[:3])

