"""
Phase 2: Enhanced NFL Injury Data Ingestion

Implements InjuryIngestion class with multi-source support:
1. nflverse (if available)
2. ESPN API (unofficial but comprehensive)
3. NFL.com scraping (fallback)

This module provides the Phase 2 interface specified in modeloptimization.md.
"""

import pandas as pd
import yaml
from pathlib import Path
from typing import List, Optional, Dict
import logging
import time
import re
from datetime import datetime, timedelta
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests library not available. API fetching will be disabled.")

# Try to import BeautifulSoup
try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    logger.warning("BeautifulSoup not available. Web scraping will be disabled.")

# Try to import nfl_data_py
try:
    import nfl_data_py as nfl
    NFLVERSE_AVAILABLE = True
except ImportError:
    NFLVERSE_AVAILABLE = False
    logger.warning("nfl_data_py not available. nflverse injury data will be disabled.")


class InjuryIngestion:
    """
    Ingest NFL injury report data from multiple sources.
    
    Injury reports are released:
    - Wednesday: First practice report
    - Thursday: Second practice report
    - Friday: Final injury designations
    - Saturday: Game-day inactives (for Sunday games)
    """
    
    def __init__(self, source: str = 'auto', cache_dir: Optional[Path] = None):
        """
        Args:
            source: 'nflverse', 'espn', 'nfl_scrape', or 'auto' (try in order)
            cache_dir: Directory for caching responses
        """
        self.source = source
        self.cache_dir = cache_dir or Path(__file__).parent.parent.parent / "data" / "nfl" / "cache" / "injuries"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config for API keys if needed
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        """Load configuration."""
        config_path = Path(__file__).parent.parent.parent / "config" / "data" / "thesportsdb.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        return {}
    
    def fetch_current_injuries(self) -> pd.DataFrame:
        """
        Fetch current week's injury report.
        
        Returns DataFrame with columns:
        - season, week
        - team (abbreviation)
        - player_id, player_name
        - position (QB, WR, RB, etc.)
        - injury_type (knee, hamstring, concussion, illness, etc.)
        - practice_status (DNP, Limited, Full)
        - game_status (Out, Doubtful, Questionable, Probable, None)
        """
        logger.info("Fetching current week's injury report...")
        
        if self.source == 'auto':
            # Try sources in order: espn, nfl_scrape, balldontlie
            for source in ['espn', 'nfl_scrape', 'balldontlie']:
                try:
                    if source == 'espn':
                        injuries = self._fetch_espn_injuries()
                        if len(injuries) > 0:
                            logger.info(f"Successfully fetched {len(injuries)} injuries from ESPN")
                            return injuries
                        continue
                    elif source == 'nfl_scrape':
                        injuries = self._scrape_nfl_injuries_all_teams()
                        if len(injuries) > 0:
                            logger.info(f"Successfully scraped {len(injuries)} injuries from NFL.com")
                            return injuries
                        continue
                    elif source == 'balldontlie':
                        # Try BALLDONTLIE API (may need API key or different endpoint)
                        try:
                            from ingestion.nfl.injuries_balldontlie import fetch_current_week_injuries
                            injuries = fetch_current_week_injuries()
                            if len(injuries) > 0:
                                logger.info(f"Successfully fetched {len(injuries)} injuries from BALLDONTLIE")
                                return injuries
                        except Exception as e:
                            logger.debug(f"BALLDONTLIE API not available: {e}")
                            continue
                except Exception as e:
                    logger.warning(f"Failed to fetch from {source}: {e}")
                    continue
            
            logger.warning("All injury data sources failed or returned no data")
            return pd.DataFrame()
        
        elif self.source == 'espn':
            return self._fetch_espn_injuries()
        elif self.source == 'nfl_scrape':
            return self._scrape_nfl_injuries_all_teams()
        elif self.source == 'balldontlie':
            from ingestion.nfl.injuries_balldontlie import fetch_current_week_injuries
            return fetch_current_week_injuries()
        elif self.source == 'nflverse':
            return self._fetch_nflverse_injuries()
        else:
            raise ValueError(f"Unknown source: {self.source}")
    
    def fetch_historical_injuries(
        self,
        seasons: List[int],
        games_df: Optional[pd.DataFrame] = None,
        use_mock_if_unavailable: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch historical injury data for backtesting.
        Need injury status AS OF game day (not after).
        
        Args:
            seasons: List of season years to fetch
            games_df: Optional games DataFrame (for mock data generation)
            use_mock_if_unavailable: Generate mock data if real sources fail
        
        Returns:
            DataFrame with historical injury data
        """
        logger.info(f"Fetching historical injuries for seasons: {seasons}")
        
        all_injuries = []
        
        for season in seasons:
            try:
                # Try ESPN first (most comprehensive)
                season_injuries = self._fetch_espn_injuries_season(season)
                if len(season_injuries) > 0:
                    all_injuries.append(season_injuries)
                    logger.info(f"Fetched {len(season_injuries)} injuries for {season} from ESPN")
                else:
                    logger.warning(f"No ESPN injury data for {season}, trying other sources...")
            except Exception as e:
                logger.warning(f"Error fetching ESPN injuries for {season}: {e}")
        
        # If no real data and mock is allowed, generate mock data
        if len(all_injuries) == 0 and use_mock_if_unavailable:
            if games_df is not None and len(games_df) > 0:
                logger.info("No real injury data available, generating mock data for validation...")
                from ingestion.nfl.injury_data_generator import generate_mock_injury_data
                # Filter games to requested seasons
                season_games = games_df[games_df['season'].isin(seasons)].copy()
                if len(season_games) > 0:
                    mock_injuries = generate_mock_injury_data(season_games)
                    all_injuries.append(mock_injuries)
                    logger.info(f"Generated {len(mock_injuries)} mock injuries for validation")
            else:
                logger.warning("No games DataFrame provided for mock data generation")
        
        if len(all_injuries) == 0:
            logger.warning("No historical injury data fetched")
            return pd.DataFrame()
        
        df = pd.concat(all_injuries, ignore_index=True)
        logger.info(f"Total historical injuries fetched: {len(df)}")
        return df
    
    def get_team_injuries(self, team: str, week: int, season: int) -> pd.DataFrame:
        """
        Get injuries for a specific team entering a specific game.
        
        Args:
            team: Team abbreviation
            week: Week number
            season: Season year
        
        Returns:
            DataFrame with injuries for that team/week
        """
        # Fetch all injuries for the season/week
        injuries = self.fetch_historical_injuries([season])
        
        if len(injuries) == 0:
            return pd.DataFrame()
        
        # Filter to team and week
        team_injuries = injuries[
            (injuries['team'] == team) &
            (injuries['week'] == week) &
            (injuries['season'] == season)
        ].copy()
        
        return team_injuries
    
    def _fetch_espn_injuries(self) -> pd.DataFrame:
        """
        ESPN injury API endpoint (unofficial):
        https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries
        
        Parse JSON response to extract injury data.
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library required for ESPN API")
        
        url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            injuries = []
            
            # Parse ESPN response structure
            # ESPN returns events (games) with teams and injuries
            if 'events' in data:
                for event in data['events']:
                    # Extract game info
                    game_date = event.get('date', '')
                    # Try to extract season/week from date or event info
                    season = self._extract_season_from_date(game_date)
                    week = self._extract_week_from_event(event)
                    
                    # Extract team injuries
                    if 'competitions' in event:
                        for competition in event['competitions']:
                            if 'competitors' in competition:
                                for competitor in competition['competitors']:
                                    team_name = competitor.get('team', {}).get('displayName', '')
                                    team_abbrev = self._map_team_name_to_abbrev(team_name)
                                    
                                    # Get injuries for this team
                                    if 'injuries' in competitor:
                                        for injury in competitor['injuries']:
                                            player = injury.get('athlete', {})
                                            injuries.append({
                                                'season': season,
                                                'week': week,
                                                'team': team_abbrev,
                                                'player_id': player.get('id', ''),
                                                'player_name': player.get('displayName', ''),
                                                'position': player.get('position', {}).get('abbreviation', ''),
                                                'injury_type': injury.get('type', ''),
                                                'practice_status': injury.get('status', ''),
                                                'game_status': injury.get('status', ''),
                                            })
            
            df = pd.DataFrame(injuries)
            logger.info(f"Fetched {len(df)} injuries from ESPN API")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching ESPN injuries: {e}")
            raise
    
    def _fetch_espn_injuries_season(self, season: int) -> pd.DataFrame:
        """Fetch injuries for a specific season from ESPN."""
        # ESPN API doesn't have a direct historical endpoint
        # Try to use existing injuries.py module as fallback
        try:
            from ingestion.nfl.injuries import fetch_nfl_injuries
            logger.info(f"Using existing injuries.py module for season {season}")
            existing_injuries = fetch_nfl_injuries([season], use_api=True, use_scraping=True)
            if len(existing_injuries) > 0:
                # Convert to Phase 2 format
                if 'game_id' in existing_injuries.columns:
                    # Map to Phase 2 schema
                    phase2_injuries = pd.DataFrame({
                        'season': existing_injuries.get('season', season),
                        'week': existing_injuries.get('week', 1),
                        'team': existing_injuries.get('team', ''),
                        'player_id': existing_injuries.get('player_id', ''),
                        'player_name': existing_injuries.get('player_name', ''),
                        'position': existing_injuries.get('position', ''),
                        'injury_type': existing_injuries.get('injury_type', ''),
                        'practice_status': existing_injuries.get('practice_status', ''),
                        'game_status': existing_injuries.get('status', ''),
                    })
                    return phase2_injuries
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"Error using existing injuries module: {e}")
            logger.warning(f"ESPN API doesn't support historical data directly. Season {season} skipped.")
            return pd.DataFrame()
    
    def _scrape_nfl_injuries_all_teams(self) -> pd.DataFrame:
        """
        Scrape official injury report from NFL.com for all teams.
        
        URL pattern: https://www.nfl.com/teams/{team}/injuries
        """
        if not BEAUTIFULSOUP_AVAILABLE or not REQUESTS_AVAILABLE:
            raise ImportError("BeautifulSoup and requests required for scraping")
        
        teams = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
                 'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
                 'LV', 'LAR', 'LAC', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
                 'NYJ', 'PHI', 'PIT', 'SF', 'SEA', 'TB', 'TEN', 'WAS']
        
        all_injuries = []
        
        for team in teams:
            try:
                team_injuries = self._scrape_nfl_injuries(team)
                if len(team_injuries) > 0:
                    all_injuries.append(team_injuries)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                logger.warning(f"Error scraping injuries for {team}: {e}")
                continue
        
        if len(all_injuries) == 0:
            return pd.DataFrame()
        
        df = pd.concat(all_injuries, ignore_index=True)
        logger.info(f"Scraped {len(df)} injuries from NFL.com")
        return df
    
    def _scrape_nfl_injuries(self, team: str) -> pd.DataFrame:
        """
        Scrape official injury report from NFL.com
        URL pattern: https://www.nfl.com/teams/{team}/injuries
        
        Try both HTML scraping and JSON API endpoints.
        """
        if not BEAUTIFULSOUP_AVAILABLE or not REQUESTS_AVAILABLE:
            raise ImportError("BeautifulSoup and requests required")
        
        # Map team abbreviation to NFL.com URL format
        team_url_map = {
            'ARI': 'cardinals', 'ATL': 'falcons', 'BAL': 'ravens', 'BUF': 'bills',
            'CAR': 'panthers', 'CHI': 'bears', 'CIN': 'bengals', 'CLE': 'browns',
            'DAL': 'cowboys', 'DEN': 'broncos', 'DET': 'lions', 'GB': 'packers',
            'HOU': 'texans', 'IND': 'colts', 'JAX': 'jaguars', 'KC': 'chiefs',
            'LV': 'raiders', 'LAR': 'rams', 'LAC': 'chargers', 'MIA': 'dolphins',
            'MIN': 'vikings', 'NE': 'patriots', 'NO': 'saints', 'NYG': 'giants',
            'NYJ': 'jets', 'PHI': 'eagles', 'PIT': 'steelers', 'SF': '49ers',
            'SEA': 'seahawks', 'TB': 'buccaneers', 'TEN': 'titans', 'WAS': 'commanders',
        }
        
        team_slug = team_url_map.get(team, team.lower())
        
        # Use the main NFL.com injuries page (https://www.nfl.com/injuries/)
        # This page has all teams' injuries in one place, organized by game
        main_url = "https://www.nfl.com/injuries/"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
        
        try:
            response = requests.get(main_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the team's section in the injuries page
            # Look for team abbreviation or team name in headings/sections
            team_sections = soup.find_all(['h2', 'h3', 'div'], string=lambda text: text and team.upper() in text.upper())
            
            # Also try finding tables near team name mentions
            all_tables = soup.find_all('table')
            team_injuries = []
            
            for table in all_tables:
                # Check if this table is for our team
                # Look for team name/abbrev in preceding text or parent elements
                parent_text = ''
                for parent in table.find_parents(['div', 'section', 'article']):
                    parent_text += parent.get_text() + ' '
                
                # Check if team name appears near this table
                if team.upper() in parent_text.upper() or team_slug in parent_text.lower():
                    injuries = self._parse_nfl_html_injuries(soup, team)
                    # Filter to only injuries from this table's context
                    table_injuries = self._parse_table_injuries(table, team)
                    team_injuries.extend(table_injuries)
            
            # If we didn't find team-specific tables, parse all tables and filter
            if len(team_injuries) == 0:
                # Parse all injuries and filter by team
                all_injuries = self._parse_nfl_html_injuries(soup, team)
                # The parsing function should handle team identification
                team_injuries = all_injuries
            
            if len(team_injuries) > 0:
                logger.info(f"Scraped {len(team_injuries)} injuries from NFL.com for {team}")
                return pd.DataFrame(team_injuries)
            
        except Exception as e:
            logger.warning(f"Error scraping NFL.com injuries for {team}: {e}")
        
        return pd.DataFrame()
    
    def _parse_table_injuries(self, table, team: str) -> List[dict]:
        """Parse injuries from a specific table."""
        injuries = []
        rows = table.find_all('tr')
        if len(rows) < 2:
            return injuries
        
        headers = [th.get_text(strip=True).upper() for th in rows[0].find_all(['th', 'td'])]
        player_idx = next((i for i, h in enumerate(headers) if 'PLAYER' in h), 0)
        pos_idx = next((i for i, h in enumerate(headers) if 'POSITION' in h), 1)
        injury_idx = next((i for i, h in enumerate(headers) if 'INJUR' in h), 2)
        practice_idx = next((i for i, h in enumerate(headers) if 'PRACTICE' in h), 3)
        game_status_idx = next((i for i, h in enumerate(headers) if 'GAME STATUS' in h or 'STATUS' in h), 4)
        
        for row in rows[1:]:
            cells = row.find_all(['td', 'th'])
            if len(cells) < 3:
                continue
            
            player_cell = cells[player_idx] if player_idx < len(cells) else cells[0]
            player_link = player_cell.find('a')
            player_name = player_link.get_text(strip=True) if player_link else player_cell.get_text(strip=True)
            
            if not player_name:
                continue
            
            position = cells[pos_idx].get_text(strip=True) if pos_idx < len(cells) else ''
            injury_type = cells[injury_idx].get_text(strip=True) if injury_idx < len(cells) else ''
            practice_status = cells[practice_idx].get_text(strip=True) if practice_idx < len(cells) else ''
            game_status = cells[game_status_idx].get_text(strip=True) if game_status_idx < len(cells) else ''
            
            if injury_type or game_status or practice_status:
                injuries.append({
                    'team': team,
                    'player_name': player_name,
                    'position': position,
                    'injury_type': injury_type,
                    'practice_status': practice_status,
                    'game_status': game_status,
                    'season': datetime.now().year if datetime.now().month >= 9 else datetime.now().year - 1,
                    'week': self._get_current_week(),
                })
        
        return injuries
    
    def _parse_nfl_json_injuries(self, data: dict, team: str) -> List[dict]:
        """Parse NFL.com JSON API response."""
        injuries = []
        
        # NFL.com JSON structure may vary
        # Try common patterns
        if isinstance(data, dict):
            # Look for injury data in various possible locations
            injury_list = data.get('injuries', []) or data.get('data', []) or data.get('players', [])
            
            for item in injury_list:
                player = item.get('player', {}) or item
                injury_record = {
                    'team': team,
                    'player_id': player.get('id') or player.get('player_id'),
                    'player_name': player.get('name') or player.get('full_name') or player.get('displayName'),
                    'position': player.get('position') or item.get('position'),
                    'injury_type': item.get('injury') or item.get('injury_type') or item.get('bodyPart'),
                    'practice_status': item.get('practice_status') or item.get('practiceStatus'),
                    'game_status': item.get('status') or item.get('game_status') or item.get('gameStatus'),
                    'season': datetime.now().year if datetime.now().month >= 9 else datetime.now().year - 1,
                    'week': item.get('week') or self._get_current_week(),
                }
                injuries.append(injury_record)
        
        return injuries
    
    def _parse_nfl_html_injuries(self, soup: BeautifulSoup, team: str) -> List[dict]:
        """Parse NFL.com HTML injury report from https://www.nfl.com/injuries/"""
        injuries = []
        
        # NFL.com structure: Tables with columns: Player, Position, Injuries, Practice Status, Game Status
        # Look for tables - they may be in various containers
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            if len(rows) < 2:  # Need at least header + 1 data row
                continue
            
            # Check if this looks like an injury table (has Position column)
            header_row = rows[0]
            headers = [th.get_text(strip=True).upper() for th in header_row.find_all(['th', 'td'])]
            
            # Check if this is an injury table
            if 'POSITION' not in headers and 'PLAYER' not in headers:
                continue
            
            # Find column indices
            player_idx = next((i for i, h in enumerate(headers) if 'PLAYER' in h), 0)
            pos_idx = next((i for i, h in enumerate(headers) if 'POSITION' in h), 1)
            injury_idx = next((i for i, h in enumerate(headers) if 'INJUR' in h), 2)
            practice_idx = next((i for i, h in enumerate(headers) if 'PRACTICE' in h), 3)
            game_status_idx = next((i for i, h in enumerate(headers) if 'GAME STATUS' in h or 'STATUS' in h), 4)
            
            # Parse data rows
            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) < 3:
                    continue
                
                # Extract player name (may be in a link)
                player_cell = cells[player_idx] if player_idx < len(cells) else cells[0]
                player_link = player_cell.find('a')
                player_name = player_link.get_text(strip=True) if player_link else player_cell.get_text(strip=True)
                
                # Only process if player name exists and not empty
                if not player_name or player_name == '':
                    continue
                
                position = cells[pos_idx].get_text(strip=True) if pos_idx < len(cells) else ''
                injury_type = cells[injury_idx].get_text(strip=True) if injury_idx < len(cells) else ''
                practice_status = cells[practice_idx].get_text(strip=True) if practice_idx < len(cells) else ''
                game_status = cells[game_status_idx].get_text(strip=True) if game_status_idx < len(cells) else ''
                
                # Only add if there's an actual injury or status
                if injury_type or game_status or practice_status:
                    injuries.append({
                        'team': team,
                        'player_name': player_name,
                        'position': position,
                        'injury_type': injury_type,
                        'practice_status': practice_status,
                        'game_status': game_status,
                        'season': datetime.now().year if datetime.now().month >= 9 else datetime.now().year - 1,
                        'week': self._get_current_week(),
                    })
        
        return injuries
    
    def _get_current_week(self) -> int:
        """Estimate current NFL week based on date."""
        now = datetime.now()
        # Rough estimate: NFL season starts early September (week 1)
        # This is a simplified calculation
        if now.month >= 9:
            # September onwards
            weeks_elapsed = (now.day // 7) + 1
            return min(weeks_elapsed, 18)
        else:
            # January-August: playoffs or off-season
            return 1
    
    def _fetch_nflverse_injuries(self) -> pd.DataFrame:
        """Fetch injuries from nflverse if available."""
        if not NFLVERSE_AVAILABLE:
            raise ImportError("nfl_data_py not available")
        
        # Check if nflverse has injury data
        # nflverse may not have injury data, so this might return empty
        logger.warning("nflverse injury data not yet implemented")
        return pd.DataFrame()
    
    def _extract_season_from_date(self, date_str: str) -> int:
        """Extract season year from date string."""
        try:
            date = pd.to_datetime(date_str)
            # NFL season spans two calendar years, use the later year
            if date.month >= 9:
                return date.year
            else:
                return date.year - 1
        except:
            return datetime.now().year
    
    def _extract_week_from_event(self, event: dict) -> int:
        """Extract week number from ESPN event data."""
        # ESPN event may have week info in various places
        # This is a simplified extraction
        return 1  # Placeholder
    
    def _map_team_name_to_abbrev(self, team_name: str) -> str:
        """Map ESPN team names to abbreviations."""
        team_map = {
            "Kansas City Chiefs": "KC",
            "Buffalo Bills": "BUF",
            "Miami Dolphins": "MIA",
            "New York Jets": "NYJ",
            "New England Patriots": "NE",
            "Baltimore Ravens": "BAL",
            "Cincinnati Bengals": "CIN",
            "Cleveland Browns": "CLE",
            "Pittsburgh Steelers": "PIT",
            "Houston Texans": "HOU",
            "Indianapolis Colts": "IND",
            "Jacksonville Jaguars": "JAX",
            "Tennessee Titans": "TEN",
            "Denver Broncos": "DEN",
            "Las Vegas Raiders": "LV",
            "Los Angeles Chargers": "LAC",
            "Dallas Cowboys": "DAL",
            "New York Giants": "NYG",
            "Philadelphia Eagles": "PHI",
            "Washington Commanders": "WAS",
            "Chicago Bears": "CHI",
            "Detroit Lions": "DET",
            "Green Bay Packers": "GB",
            "Minnesota Vikings": "MIN",
            "Atlanta Falcons": "ATL",
            "Carolina Panthers": "CAR",
            "New Orleans Saints": "NO",
            "Tampa Bay Buccaneers": "TB",
            "Arizona Cardinals": "ARI",
            "Los Angeles Rams": "LAR",
            "San Francisco 49ers": "SF",
            "Seattle Seahawks": "SEA",
        }
        return team_map.get(team_name, team_name[:3].upper())


# Convenience function for backward compatibility
def fetch_nfl_injuries_phase2(
    seasons: Optional[List[int]] = None,
    source: str = 'auto',
) -> pd.DataFrame:
    """
    Convenience function to fetch injuries using Phase 2 interface.
    
    Args:
        seasons: List of seasons to fetch (None = current week only)
        source: Data source ('auto', 'espn', 'nfl_scrape')
    
    Returns:
        DataFrame with injury data
    """
    ingester = InjuryIngestion(source=source)
    
    if seasons:
        return ingester.fetch_historical_injuries(seasons)
    else:
        return ingester.fetch_current_injuries()

