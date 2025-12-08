"""
Scraper for NFL Next Gen Stats API.
"""

import logging
from typing import List, Dict, Optional
import requests
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


class NextGenStatsScraper:
    """
    Scrape NFL Next Gen Stats using the public API.
    
    Base URL: https://nextgenstats.nfl.com/api/leaders/{metric}/{category}
    
    Working endpoints:
    - speed/ballCarrier - Fastest Ball Carriers
    - distance/ballCarrier - Longest Ball Carrier Runs
    - distance/tackle - Longest Tackles
    - time/sack - Fastest Sacks (by time)
    - expectation/completion/season - Improbable Completions
    - expectation/yac/season - YAC (Yards After Catch)
    - expectation/ery/season - Remarkable Rushes (Expected Rush Yards)
    """
    
    BASE_URL = "https://nextgenstats.nfl.com/api/leaders"
    
    # Available endpoints
    # Format: (metric, category, suffix)
    # suffix is None for week-based endpoints, 'season' for expectation endpoints
    ENDPOINTS = {
        'fastest-ball-carriers': ('speed', 'ballCarrier', None),
        'longest-ball-carrier-runs': ('distance', 'ballCarrier', None),
        'longest-tackles': ('distance', 'tackle', None),
        'fastest-sacks': ('time', 'sack', None),
        'improbable-completions': ('expectation', 'completion', 'season'),
        'yac': ('expectation', 'yac', 'season'),
        'remarkable-rushes': ('expectation', 'ery', 'season'),
    }
    
    def __init__(self):
        self.headers = {
            'Accept': 'application/json, text/plain, */*',
            'Origin': 'https://nextgenstats.nfl.com',
            'Referer': 'https://nextgenstats.nfl.com/',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def get_leaders(
        self,
        category: str,
        season: int,
        season_type: str = 'REG',
        week: Optional[int] = None,
        limit: int = 20
    ) -> Optional[Dict]:
        """
        Get Next Gen Stats leaders for a category.
        
        Args:
            category: One of 'fastest-ball-carriers', 'longest-ball-carrier-runs',
                     'longest-tackles', 'fastest-sacks', 'improbable-completions',
                     'yac', 'remarkable-rushes'
            season: NFL season year (2018+)
            season_type: 'REG' or 'POST'
            week: Week number (1-18) or None for all weeks (only for non-season endpoints)
            limit: Number of records to return
            
        Returns:
            Dict with 'season', 'seasonType', 'week' (if applicable), 'leaders' keys
        """
        if category not in self.ENDPOINTS:
            logger.error(f"Unknown category: {category}")
            return None
        
        metric, endpoint_category, suffix = self.ENDPOINTS[category]
        
        # Build URL based on endpoint type
        if suffix:
            # Expectation endpoints use /season suffix
            url = f"{self.BASE_URL}/{metric}/{endpoint_category}/{suffix}"
        else:
            # Standard endpoints
            url = f"{self.BASE_URL}/{metric}/{endpoint_category}"
        
        params = {
            'limit': limit,
            'season': season,
            'seasonType': season_type
        }
        
        # Week parameter only applies to non-season endpoints
        if suffix is None and week is not None:
            params['week'] = week
        elif suffix == 'season' and week is not None:
            # For season endpoints, week can still be used to filter
            params['week'] = week
        
        try:
            logger.info(f"Fetching Next Gen Stats: {category}, season={season}, week={week}")
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return data
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error fetching Next Gen Stats: {e}")
            if e.response.status_code == 403:
                logger.error("This endpoint may require AWS signature authentication")
            return None
        except Exception as e:
            logger.error(f"Error fetching Next Gen Stats: {e}")
            return None
    
    def extract_leader_data(self, leader_record: Dict) -> Dict:
        """
        Extract key data from a leader record.
        
        Returns flattened dict with key metrics.
        """
        # All endpoints use nested structure with 'leader' and 'play'
        leader = leader_record.get('leader', {})
        play = leader_record.get('play', {})
        
        result = {
            'player_id': leader.get('gsisId'),
            'player_name': leader.get('playerName'),
            'team': leader.get('teamAbbr'),
            'position': leader.get('position'),
            'jersey_number': leader.get('jerseyNumber'),
            'week': leader.get('week'),
            
            # Metrics (varies by category)
            'max_speed': leader.get('maxSpeed'),  # For speed endpoints
            'yards': leader.get('yards'),  # For distance endpoints
            'time': leader_record.get('time'),  # For time endpoints
            'in_play_dist': leader.get('inPlayDist'),
            
            # Expectation-based metrics (for expectation endpoints)
            'expectation': leader_record.get('expectation') or leader.get('completionProbability'),
            'actual': leader_record.get('actual'),
            'difference': leader_record.get('difference'),
            'completion_probability': leader.get('completionProbability'),  # For improbable completions
            'air_yards': leader.get('airYards'),  # For completions
            'pass_yards': leader.get('passYards'),  # For completions
            'yac': leader.get('yac'),  # For YAC endpoint
            'expected_yac': leader.get('expectedYac'),  # For YAC endpoint
            'expected_rush_yards': leader.get('expectedRushYards'),  # For remarkable rushes
            'rush_yards': leader.get('rushYards'),  # For remarkable rushes
            
            # Play info
            'game_id': play.get('gameId'),
            'play_id': play.get('playId'),
            'down': play.get('down'),
            'game_clock': play.get('gameClock'),
            'is_big_play': play.get('isBigPlay'),
        }
        
        return result
    
    def scrape_category(
        self,
        category: str,
        season: int,
        season_type: str = 'REG',
        week: Optional[int] = None,
        limit: int = 20
    ) -> List[Dict]:
        """
        Scrape a category and return list of extracted records.
        
        Returns list of flattened dicts.
        """
        data = self.get_leaders(category, season, season_type, week, limit)
        
        if not data:
            return []
        
        # Different endpoints use different keys for leaders
        leaders_key = None
        if 'leaders' in data:
            leaders_key = 'leaders'
        elif 'completionLeaders' in data:
            leaders_key = 'completionLeaders'
        elif 'yacLeaders' in data:
            leaders_key = 'yacLeaders'
        elif 'eryLeaders' in data:
            leaders_key = 'eryLeaders'
        
        if not leaders_key:
            logger.warning(f"No leaders found in response for {category}")
            return []
        
        records = []
        for leader_record in data[leaders_key]:
            extracted = self.extract_leader_data(leader_record)
            extracted['category'] = category
            extracted['season'] = season
            extracted['season_type'] = season_type
            records.append(extracted)
        
        return records
    
    def scrape_all_categories(
        self,
        season: int,
        season_type: str = 'REG',
        week: Optional[int] = None,
        limit: int = 20
    ) -> Dict[str, List[Dict]]:
        """
        Scrape all available categories.
        
        Returns dict mapping category name to list of records.
        """
        results = {}
        
        for category in self.ENDPOINTS.keys():
            logger.info(f"Scraping category: {category}")
            records = self.scrape_category(category, season, season_type, week, limit)
            results[category] = records
            logger.info(f"  Extracted {len(records)} records")
        
        return results

