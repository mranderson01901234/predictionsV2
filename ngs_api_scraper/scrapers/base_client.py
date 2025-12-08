"""
Base API client for Next Gen Stats.

Handles:
- Rate limiting
- Request retries with exponential backoff
- Response caching
- Error handling
- Common headers
"""

import time
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RateLimiter:
    """Token bucket rate limiter."""
    
    requests_per_second: float
    last_request: float = 0
    
    def wait(self):
        """Wait if necessary to respect rate limit."""
        min_interval = 1.0 / self.requests_per_second
        elapsed = time.time() - self.last_request
        
        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            time.sleep(sleep_time)
        
        self.last_request = time.time()


class ResponseCache:
    """File-based JSON response cache."""
    
    def __init__(
        self, 
        cache_dir: str = "data/cache",
        default_expiry_hours: int = 1,
        historical_expiry_days: int = 30
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_expiry = timedelta(hours=default_expiry_hours)
        self.historical_expiry = timedelta(days=historical_expiry_days)
    
    def _get_cache_key(self, url: str, params: Dict) -> str:
        """Generate cache key from URL and params."""
        key_str = f"{url}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{cache_key}.json"
    
    def get(
        self, 
        url: str, 
        params: Dict,
        is_historical: bool = False
    ) -> Optional[Dict]:
        """Get cached response if exists and not expired."""
        cache_key = self._get_cache_key(url, params)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        # Check expiry
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        expiry = self.historical_expiry if is_historical else self.default_expiry
        
        if datetime.now() - mtime > expiry:
            logger.debug(f"Cache expired for {url}")
            return None
        
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None
    
    def set(self, url: str, params: Dict, data: Dict):
        """Cache response data."""
        cache_key = self._get_cache_key(url, params)
        cache_path = self._get_cache_path(cache_key)
        
        with open(cache_path, 'w') as f:
            json.dump(data, f)


class NGSClient:
    """
    Next Gen Stats API client.
    
    Usage:
        client = NGSClient()
        
        # Get passing stats
        passing = client.get_statboard('passing', season=2024, season_type='REG')
        
        # Get rushing stats for specific week
        rushing = client.get_statboard('rushing', season=2024, season_type='REG', week=10)
        
        # Get game center data
        game = client.get_game_center('2024120100')
    """
    
    BASE_URL = "https://nextgenstats.nfl.com/api"
    
    # Required headers (API rejects without these)
    DEFAULT_HEADERS = {
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Origin': 'https://nextgenstats.nfl.com',
        'Referer': 'https://nextgenstats.nfl.com/',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
    }
    
    def __init__(
        self,
        requests_per_second: float = 2.0,
        max_retries: int = 3,
        cache_enabled: bool = True,
        cache_dir: str = "data/cache"
    ):
        self.rate_limiter = RateLimiter(requests_per_second)
        self.cache = ResponseCache(cache_dir) if cache_enabled else None
        
        # Setup session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        
        # Stats
        self.requests_made = 0
        self.cache_hits = 0
    
    def _request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        use_cache: bool = True,
        is_historical: bool = False
    ) -> Optional[Dict]:
        """
        Make API request with rate limiting and caching.
        
        Args:
            endpoint: API endpoint path (e.g., '/statboard/passing')
            params: Query parameters
            use_cache: Whether to use cache
            is_historical: Use longer cache expiry for historical data
            
        Returns:
            JSON response as dict, or None if failed
        """
        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}
        
        # Check cache
        if use_cache and self.cache:
            cached = self.cache.get(url, params, is_historical)
            if cached:
                self.cache_hits += 1
                logger.debug(f"Cache hit: {endpoint}")
                return cached
        
        # Rate limit
        self.rate_limiter.wait()
        
        try:
            logger.info(f"Requesting: {endpoint} {params}")
            
            response = self.session.get(
                url,
                params=params,
                headers=self.DEFAULT_HEADERS,
                timeout=30
            )
            
            self.requests_made += 1
            
            if response.status_code == 200:
                data = response.json()
                
                # Cache response
                if self.cache:
                    self.cache.set(url, params, data)
                
                return data
            else:
                logger.error(f"HTTP {response.status_code}: {response.text[:200]}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {endpoint}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {endpoint}: {e}")
            return None
    
    # === STATBOARD ENDPOINTS ===
    
    def get_statboard(
        self,
        stat_type: str,
        season: int,
        season_type: str = 'REG',
        week: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Get player statistics board.
        
        Args:
            stat_type: 'passing', 'rushing', or 'receiving'
            season: Season year (2018+)
            season_type: 'REG' or 'POST'
            week: Week number (optional, omit for season totals)
            
        Returns:
            Dict with 'stats' list containing player data
        """
        if stat_type not in ['passing', 'rushing', 'receiving']:
            raise ValueError(f"Invalid stat_type: {stat_type}")
        
        endpoint = f"/statboard/{stat_type}"
        params = {
            'season': season,
            'seasonType': season_type,
        }
        
        if week is not None:
            params['week'] = week
        
        is_historical = season < datetime.now().year
        return self._request(endpoint, params, is_historical=is_historical)
    
    def get_passing_stats(
        self,
        season: int,
        season_type: str = 'REG',
        week: Optional[int] = None
    ) -> Optional[List[Dict]]:
        """Get passing statistics. Returns list of player stats."""
        data = self.get_statboard('passing', season, season_type, week)
        return data.get('stats', []) if data else None
    
    def get_rushing_stats(
        self,
        season: int,
        season_type: str = 'REG',
        week: Optional[int] = None
    ) -> Optional[List[Dict]]:
        """Get rushing statistics. Returns list of player stats."""
        data = self.get_statboard('rushing', season, season_type, week)
        return data.get('stats', []) if data else None
    
    def get_receiving_stats(
        self,
        season: int,
        season_type: str = 'REG',
        week: Optional[int] = None
    ) -> Optional[List[Dict]]:
        """Get receiving statistics. Returns list of player stats."""
        data = self.get_statboard('receiving', season, season_type, week)
        return data.get('stats', []) if data else None
    
    # === LEADERS ENDPOINTS ===
    
    def get_leaders(
        self,
        leader_type: str,
        season: int,
        season_type: str = 'REG',
        week: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Get top plays leaders.
        
        Args:
            leader_type: One of:
                - 'speed/ballCarrier' (fastest ball carriers)
                - 'distance/ballCarrier' (longest runs)
                - 'distance/tackle' (longest tackles)
                - 'time/sack' (fastest sacks)
                - 'expectation/completion/season' (improbable completions)
                - 'expectation/yac/season' (YAC above expected)
                - 'expectation/ery/season' (remarkable rushes)
            season: Season year
            season_type: 'REG' or 'POST'
            week: Week number (optional)
        """
        endpoint = f"/leaders/{leader_type}"
        params = {
            'season': season,
            'seasonType': season_type,
        }
        if week:
            params['week'] = week
        
        return self._request(endpoint, params, is_historical=season < datetime.now().year)
    
    # === GAME CENTER ===
    
    def get_game_center(self, game_id: str) -> Optional[Dict]:
        """
        Get detailed stats for a specific game.
        
        Args:
            game_id: Game ID in format YYYYMMDDHH (e.g., '2024120100')
            
        Returns:
            Dict with passers, rushers, receivers, pass rushers, leaders
        """
        endpoint = "/gamecenter/overview"
        params = {'gameId': game_id}
        
        # Game center data is historical once game is complete
        return self._request(endpoint, params, is_historical=True)
    
    # === HIGHLIGHTS ===
    
    def get_highlights(
        self,
        season: int,
        season_type: str = 'REG',
        week: Optional[int] = None,
        limit: int = 100
    ) -> Optional[Dict]:
        """
        Get highlight plays.
        
        Args:
            season: Season year
            season_type: 'REG' or 'POST'
            week: Week number (optional)
            limit: Number of highlights to return
            
        Returns:
            Dict with 'highlights' list
        """
        endpoint = "/plays/highlights"
        params = {
            'season': season,
            'seasonType': season_type,
            'limit': limit,
        }
        if week:
            params['week'] = week
        
        return self._request(endpoint, params, is_historical=season < datetime.now().year)
    
    # === CHARTS ===
    
    def get_charts(
        self,
        season: int,
        chart_type: str = 'all',
        season_type: str = 'REG',
        week: Optional[int] = None,
        team_id: str = 'all',
        player_id: str = 'all',
        count: int = 100
    ) -> Optional[Dict]:
        """
        Get pass/route/carry charts.
        
        Args:
            season: Season year
            chart_type: 'pass', 'route', 'carry', or 'all'
            season_type: 'REG' or 'POST'
            week: Week number (optional)
            team_id: Team ID or 'all'
            player_id: Player ESB ID or 'all'
            count: Number of charts to return
            
        Returns:
            Dict with 'charts' list
        """
        endpoint = "/content/microsite/chart"
        params = {
            'season': season,
            'seasonType': season_type,
            'type': chart_type,
            'teamId': team_id,
            'esbId': player_id,
            'count': count,
            'week': week if week else 'all',
        }
        
        return self._request(endpoint, params, is_historical=season < datetime.now().year)
    
    # === UTILITIES ===
    
    def get_stats(self) -> Dict[str, int]:
        """Get client statistics."""
        return {
            'requests_made': self.requests_made,
            'cache_hits': self.cache_hits,
        }


# Quick test
if __name__ == "__main__":
    client = NGSClient()
    
    # Test passing stats
    passing = client.get_passing_stats(2024, 'REG')
    if passing:
        print(f"Got {len(passing)} passers")
        print(f"Top passer: {passing[0].get('playerName')}")
        print(f"CPOE: {passing[0].get('completionPercentageAboveExpectation')}")
    
    print(f"\nClient stats: {client.get_stats()}")



