"""
Scraper for NFL.com standings data.
"""

import logging
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers.api_scraper import NFLAPIScraper

logger = logging.getLogger(__name__)


class StandingsScraper(NFLAPIScraper):
    """
    Scrape NFL standings using API.
    
    Standings endpoint provides:
    - Overall record (W-L-T)
    - Home/road splits
    - Points for/against
    - Last 5 games
    - Division/conference rank
    - Win/loss streaks
    - Close games record
    """
    
    def get_standings(
        self,
        season: int,
        season_type: str = 'REG',
        week: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Get standings for a season.
        
        Args:
            season: NFL season year
            season_type: 'REG' or 'POST'
            week: Optional week number (if None, returns all weeks)
            
        Returns:
            Dict with weeks and standings data
        """
        url = f"{self.FOOTBALL_API_BASE_URL}/standings"
        params = {
            'season': season,
            'seasonType': season_type
        }
        
        if week:
            params['week'] = week
        
        # Use fetch_api but with football base URL
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        full_url = f"{url}?{query_string}"
        
        # Check cache
        if self.cache:
            cached = self._get_cached_json(full_url)
            if cached:
                self.cache_hits += 1
                return cached
        
        # Rate limit
        self.rate_limiter.wait()
        
        try:
            logger.info(f"Fetching standings: season={season}, type={season_type}")
            response = self.session.get(full_url, timeout=self.config['request']['timeout'])
            response.raise_for_status()
            
            self.requests_made += 1
            
            # Handle Brotli compression
            content_encoding = response.headers.get('Content-Encoding', '')
            if content_encoding == 'br':
                try:
                    import brotli
                    decompressed = brotli.decompress(response.content)
                    data = json.loads(decompressed)
                except ImportError:
                    data = response.json()
                except Exception:
                    data = response.json()
            else:
                data = response.json()
            
            # Cache response
            if self.cache:
                self._cache_json(full_url, data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching standings: {e}")
            self.errors += 1
            return None
    
    def get_week_standings(
        self,
        season: int,
        week: int,
        season_type: str = 'REG'
    ) -> Optional[List[Dict]]:
        """
        Get standings for a specific week.
        
        Returns list of team standings for that week.
        """
        data = self.get_standings(season, season_type, week)
        
        if data and 'weeks' in data:
            for week_data in data['weeks']:
                if week_data.get('week') == week:
                    return week_data.get('standings', [])
        
        return None
    
    def extract_team_stats(self, standing: Dict) -> Dict:
        """
        Extract key team stats from standing record.
        
        Returns flattened dict with key metrics.
        """
        team_name = standing['team']['fullName']
        team_id = standing['team']['id']
        
        overall = standing['overall']
        home = standing['home']
        road = standing['road']
        last5 = standing['last5']
        division = standing['division']
        conference = standing['conference']
        
        return {
            'team_id': team_id,
            'team_name': team_name,
            'season': standing.get('season', None),
            'week': standing.get('week', None),
            
            # Overall
            'wins': overall['wins'],
            'losses': overall['losses'],
            'ties': overall.get('ties', 0),
            'win_pct': overall.get('winPct', 0.0),
            'points_for': overall['points']['for'],
            'points_against': overall['points']['against'],
            'point_differential': overall['points']['for'] - overall['points']['against'],
            'streak_type': overall['streak']['type'],
            'streak_length': overall['streak']['length'],
            
            # Home
            'home_wins': home['wins'],
            'home_losses': home['losses'],
            'home_win_pct': home.get('winPct', 0.0),
            'home_points_for': home['points']['for'],
            'home_points_against': home['points']['against'],
            
            # Road
            'road_wins': road['wins'],
            'road_losses': road['losses'],
            'road_win_pct': road.get('winPct', 0.0),
            'road_points_for': road['points']['for'],
            'road_points_against': road['points']['against'],
            
            # Last 5
            'last5_wins': last5['wins'],
            'last5_losses': last5['losses'],
            'last5_win_pct': last5.get('winPct', 0.0),
            'last5_points_for': last5['points']['for'],
            'last5_points_against': last5['points']['against'],
            
            # Division
            'division_rank': division['rank'],
            'division_wins': division['wins'],
            'division_losses': division['losses'],
            'division_win_pct': division.get('winPct', 0.0),
            
            # Conference
            'conference_rank': conference['rank'],
            'conference_wins': conference['wins'],
            'conference_losses': conference['losses'],
            'conference_win_pct': conference.get('winPct', 0.0),
            
            # Close games
            'close_games_wins': standing['closeGames']['wins'],
            'close_games_losses': standing['closeGames']['losses'],
        }

