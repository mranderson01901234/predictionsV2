"""
API-based injury scraper using NFL.com API endpoints.
"""

import logging
from typing import List, Optional
from datetime import datetime, date

from scrapers.api_scraper import NFLAPIScraper
from storage.schemas import InjuryRecord

logger = logging.getLogger(__name__)


class APIInjuryScraper(NFLAPIScraper):
    """
    Scrape injury reports using NFL.com API.
    
    This is more reliable than HTML scraping since it uses the official API.
    """
    
    def scrape_week(
        self,
        season: int,
        week: int,
        use_cache: bool = True
    ) -> List[InjuryRecord]:
        """
        Scrape injury report for a specific week using API.
        
        Args:
            season: NFL season year
            week: Week number
            use_cache: Whether to use cached responses
            
        Returns:
            List of InjuryRecord objects
        """
        # Get injuries from API
        api_data = self.get_injuries(season, week)
        
        if not api_data:
            logger.warning(f"No injury data from API for {season} week {week}")
            return []
        
        # Parse API response into InjuryRecord objects
        records = []
        
        # API returns list of injury records
        if isinstance(api_data, list):
            for item in api_data:
                record = self._parse_api_injury(item, season, week)
                if record:
                    records.append(record)
        elif isinstance(api_data, dict):
            # Handle nested structure (shouldn't happen with current endpoint)
            injuries = api_data.get('injuries', [])
            for injury in injuries:
                record = self._parse_api_injury(injury, season, week)
                if record:
                    records.append(record)
        
        logger.info(f"Parsed {len(records)} injury records from API for {season} week {week}")
        return records
    
    def _parse_api_injury(
        self,
        injury_data: dict,
        season: int,
        week: int,
        game_data: Optional[dict] = None
    ) -> Optional[InjuryRecord]:
        """
        Parse a single injury record from API response.
        
        API structure:
        {
          "season": 2024,
          "week": 1,
          "team": {"id": "...", "fullName": "Baltimore Ravens"},
          "person": {"id": "...", "displayName": "Zay Flowers", ...},
          "injuries": ["Knee"],
          "injuryStatus": "OUT",
          "practiceDays": [{"date": "2025-01-07", "status": "DIDNOT"}, ...],
          "practiceStatus": "DIDNOT",
          "position": "WR"
        }
        """
        try:
            # Extract player info
            person = injury_data.get('person', {})
            player_name = person.get('displayName', '')
            player_id = person.get('id', '') or person.get('gsisId', '')
            
            # Extract injury info
            injuries_list = injury_data.get('injuries', [])
            injury_types = injuries_list if isinstance(injuries_list, list) else []
            injury_type = injury_types[0] if injury_types else None
            
            # Extract practice status (DIDNOT, LIMITED, FULL)
            practice_status_raw = injury_data.get('practiceStatus', '')
            practice_status = self._normalize_practice_status(practice_status_raw)
            
            # Extract practice days (Wed/Thu/Fri)
            practice_days = injury_data.get('practiceDays', [])
            practice_status_wed = None
            practice_status_thu = None
            practice_status_fri = None
            
            for day in practice_days:
                date_str = day.get('date', '')
                status = day.get('status', '')
                normalized_status = self._normalize_practice_status(status)
                
                # Determine day of week from date
                try:
                    day_date = datetime.fromisoformat(date_str).date()
                    day_of_week = day_date.weekday()  # 0=Monday, 2=Wednesday, 3=Thursday, 4=Friday
                    
                    if day_of_week == 2:  # Wednesday
                        practice_status_wed = normalized_status
                    elif day_of_week == 3:  # Thursday
                        practice_status_thu = normalized_status
                    elif day_of_week == 4:  # Friday
                        practice_status_fri = normalized_status
                except:
                    pass
            
            # Extract game status
            game_status_raw = injury_data.get('injuryStatus', '')
            game_status = self._normalize_game_status(game_status_raw)
            
            # Extract team info
            team = injury_data.get('team', {})
            team_name = team.get('fullName', '')
            # Convert team name to abbreviation
            team_abbr = self._team_name_to_abbr(team_name)
            
            # Extract date from practice days or use None
            game_date = None
            if practice_days:
                try:
                    last_practice_date = practice_days[-1].get('date')
                    if last_practice_date:
                        game_date = datetime.fromisoformat(last_practice_date).date()
                except:
                    pass
            
            # Check if resting (not injury related)
            is_resting = False
            practices_list = injury_data.get('practices', [])
            if practices_list and 'not injury related' in str(practices_list).lower():
                is_resting = True
            
            record = InjuryRecord(
                season=season,
                week=week,
                game_date=game_date,
                team=team_abbr,
                opponent='',  # Would need game data to determine opponent
                player_name=player_name,
                player_id=str(player_id),
                position=injury_data.get('position', ''),
                injury_type=injury_type,
                injury_types=injury_types,
                practice_status_wed=practice_status_wed,
                practice_status_thu=practice_status_thu,
                practice_status_fri=practice_status_fri,
                practice_status_final=practice_status,
                game_status=game_status,
                is_resting=is_resting,
            )
            
            return record
            
        except Exception as e:
            logger.warning(f"Error parsing API injury record: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _normalize_practice_status(self, status: str) -> str:
        """Normalize practice status from API format."""
        status_map = {
            'DIDNOT': 'DNP',
            'LIMITED': 'Limited',
            'FULL': 'Full',
            'DNP': 'DNP',
        }
        return status_map.get(status.upper(), status)
    
    def _normalize_game_status(self, status: str) -> Optional[str]:
        """Normalize game status."""
        if not status:
            return None
        status_map = {
            'OUT': 'Out',
            'DOUBTFUL': 'Doubtful',
            'QUESTIONABLE': 'Questionable',
            'PROBABLE': 'Probable',
        }
        return status_map.get(status.upper(), status)
    
    def _team_name_to_abbr(self, team_name: str) -> str:
        """Convert team full name to abbreviation."""
        team_map = {
            'Arizona Cardinals': 'ARI',
            'Atlanta Falcons': 'ATL',
            'Baltimore Ravens': 'BAL',
            'Buffalo Bills': 'BUF',
            'Carolina Panthers': 'CAR',
            'Chicago Bears': 'CHI',
            'Cincinnati Bengals': 'CIN',
            'Cleveland Browns': 'CLE',
            'Dallas Cowboys': 'DAL',
            'Denver Broncos': 'DEN',
            'Detroit Lions': 'DET',
            'Green Bay Packers': 'GB',
            'Houston Texans': 'HOU',
            'Indianapolis Colts': 'IND',
            'Jacksonville Jaguars': 'JAX',
            'Kansas City Chiefs': 'KC',
            'Las Vegas Raiders': 'LV',
            'Los Angeles Chargers': 'LAC',
            'Los Angeles Rams': 'LAR',
            'Miami Dolphins': 'MIA',
            'Minnesota Vikings': 'MIN',
            'New England Patriots': 'NE',
            'New Orleans Saints': 'NO',
            'New York Giants': 'NYG',
            'New York Jets': 'NYJ',
            'Philadelphia Eagles': 'PHI',
            'Pittsburgh Steelers': 'PIT',
            'San Francisco 49ers': 'SF',
            'Seattle Seahawks': 'SEA',
            'Tampa Bay Buccaneers': 'TB',
            'Tennessee Titans': 'TEN',
            'Washington Commanders': 'WAS',
        }
        return team_map.get(team_name, team_name[:3].upper())
    
    def scrape_season(
        self,
        season: int,
        weeks: Optional[List[int]] = None,
        use_cache: bool = True
    ) -> List[InjuryRecord]:
        """
        Scrape injury reports for an entire season.
        """
        if weeks is None:
            weeks = list(range(1, 19))
        
        all_records = []
        
        for week in weeks:
            logger.info(f"Scraping injuries via API: {season} week {week}")
            
            try:
                records = self.scrape_week(season, week, use_cache)
                all_records.extend(records)
                logger.info(f"  Found {len(records)} records")
            except Exception as e:
                logger.error(f"Error scraping {season} week {week}: {e}")
                continue
        
        return all_records

