"""
API-based scraper for NFL.com using their official API endpoints.

This is more reliable than HTML scraping since it uses the same API
that the website uses internally.
"""

import logging
import json
import time
import random
from typing import Optional, Dict, Any, List
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yaml

from scrapers.base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class NFLAPIScraper(BaseScraper):
    """
    Scraper for NFL.com API endpoints.
    
    Uses the official NFL.com API at api.nfl.com/experience/v1/
    Requires authorization token (may need to be refreshed periodically).
    """
    
    API_BASE_URL = "https://api.nfl.com/experience/v1"
    FOOTBALL_API_BASE_URL = "https://api.nfl.com/football/v2"
    
    def __init__(self, config_path: str = "config/scraping_config.yaml", auth_token: Optional[str] = None):
        """
        Initialize API scraper.
        
        Args:
            config_path: Path to config file
            auth_token: Authorization Bearer token (if None, will try to get from config or generate)
        """
        super().__init__(config_path)
        
        # Load or set auth token
        self.auth_token = auth_token or self._get_auth_token()
        
        # Override session headers for API
        self.session.headers.update({
            'Accept': 'application/json, text/plain, */*',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Accept-Language': 'en-US,en;q=0.9',
            'Origin': 'https://www.nfl.com',
            'Referer': 'https://www.nfl.com/',
            'Authorization': f'Bearer {self.auth_token}',
        })
    
    def _get_auth_token(self) -> str:
        """
        Get authorization token from config or generate a default one.
        
        Note: Tokens expire. You may need to:
        1. Extract a fresh token from browser DevTools
        2. Store it in config/credentials.yaml
        3. Or implement token refresh logic
        """
        # Try to load from credentials file
        creds_path = Path(__file__).parent.parent / "config" / "credentials.yaml"
        if creds_path.exists():
            try:
                with open(creds_path) as f:
                    creds = yaml.safe_load(f)
                    if creds and 'nfl_api' in creds and 'auth_token' in creds['nfl_api']:
                        return creds['nfl_api']['auth_token']
            except Exception as e:
                logger.warning(f"Could not load auth token from credentials: {e}")
        
        # Default token (may expire - user should update)
        # This is a placeholder - user should extract fresh token from browser
        default_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJjbGllbnRJZCI6ImU1MzVjN2MwLTgxN2YtNDc3Ni04OTkwLTU2NTU2ZjhiMTkyOCIsImNsaWVudEtleSI6IjRjRlVXNkRtd0pwelQ5TDdMckczcVJBY0FCRzVzMDRnIiwiZGV2aWNlSWQiOiJmMWY1ZDEyNy0zZDg0LTRhNjktYTBlNi1lM2Q1NTNkYjIxNTIiLCJpc3MiOiJORkwiLCJwbGFucyI6W3sicGxhbiI6ImZyZWUiLCJleHBpcmF0aW9uRGF0ZSI6IjIwMjYtMTItMDgiLCJzb3VyY2UiOiJORkwiLCJzdGFydERhdGUiOiIyMDI1LTEyLTA4Iiwic3RhdHVzIjoiQUNUSVZFIiwidHJpYWwiOmZhbHNlfV0sIkRpc3BsYXlOYW1lIjoiV0VCX0RFU0tUT1BfREVTS1RPUCIsIk5vdGVzIjoiIiwiZm9ybUZhY3RvciI6IkRFU0tUT1AiLCJsdXJhQXBwS2V5IjoiU1pzNTdkQkdSeGJMNzI4bFZwN0RZUSIsInBsYXRmb3JtIjoiREVTS1RPUCIsInByb2R1Y3ROYW1lIjoiV0VCIiwicm9sZXMiOlsiY29udGVudCIsImV4cGVyaWVuY2UiLCJmb290YmFsbCIsInV0aWxpdGllcyIsInRlYW1zIiwicGxheSIsImxpdmUiLCJpZGVudGl0eSIsIm5nc19zdGF0cyIsInBheW1lbnRzX2FwaSIsIm5nc190cmFja2luZyIsIm5nc19wbGF0Zm9ybSIsIm5nc19jb250ZW50IiwibmdzX2NvbWJpbmUiLCJuZ3NfYWR2YW5jZWRfc3RhdHMiLCJuZmxfcHJvIiwiZWNvbW0iLCJuZmxfaWRfYXBpIiwidXRpbGl0aWVzX2xvY2F0aW9uIiwiaWRlbnRpdHlfb2lkYyIsIm5nc19zc2UiLCJhY2NvdW50cyIsImNvbnNlbnRzIiwic3ViX3BhcnRuZXJzaGlwcyIsImNvbmN1cnJlbmN5Iiwia2V5c3RvcmUiLCJpZF9zZXJ2ZXJfdG9fc2VydmVyIiwiZnJlZSJdLCJuZXR3b3JrVHlwZSI6Im90aGVyIiwiY2l0eSI6ImF0bGFudGEiLCJjb3VudHJ5Q29kZSI6IlVTIiwiZG1hQ29kZSI6IjUyNCIsImhtYVRlYW1zIjpbIjEwNDAwMjAwLWY0MDEtNGU1My01MTc1LTA5NzRlNGYxNmNmNyJdLCJyZWdpb24iOiJHQSIsInppcENvZGUiOiIzMDMwOSIsImJyb3dzZXIiOiJDaHJvbWUiLCJjZWxsdWxhciI6ZmFsc2UsImVudmlyb25tZW50IjoicHJvZHVjdGlvbiIsImV4cCI6MTc2NTE1OTM5OH0.PoMtMSTeRbou9KgG5ySHtcEUtRsefpXbSdGSMOCHRlA"
        
        logger.warning(
            "Using default auth token. This may expire. "
            "Extract a fresh token from browser DevTools and add to config/credentials.yaml"
        )
        return default_token
    
    def fetch_api(self, endpoint: str, params: Optional[Dict] = None, use_cache: bool = True) -> Optional[Dict]:
        """
        Fetch data from NFL.com API.
        
        Args:
            endpoint: API endpoint (e.g., "teams", "injuries", "players/{id}/stats")
            params: Query parameters
            use_cache: Whether to use cache
            
        Returns:
            JSON response as dict, or None if failed
        """
        # Construct URL
        if endpoint.startswith('/'):
            url = f"{self.API_BASE_URL}{endpoint}"
        else:
            url = f"{self.API_BASE_URL}/{endpoint}"
        
        # Add query parameters
        if params:
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            url = f"{url}?{query_string}"
        
        # Check cache (use JSON cache for API responses)
        if use_cache and self.cache:
            cached = self._get_cached_json(url)
            if cached:
                self.cache_hits += 1
                return cached
        
        # Rate limit
        self.rate_limiter.wait()
        
        try:
            logger.info(f"Fetching API: {endpoint}")
            response = self.session.get(url, timeout=self.config['request']['timeout'])
            response.raise_for_status()
            
            self.requests_made += 1
            
            # Handle Brotli compression if needed
            content_encoding = response.headers.get('Content-Encoding', '')
            if content_encoding == 'br':
                try:
                    import brotli
                    decompressed = brotli.decompress(response.content)
                    data = json.loads(decompressed)
                except ImportError:
                    logger.warning("brotli library not installed. Install with: pip install brotli")
                    # Try to use requests automatic decompression
                    data = response.json()
                except Exception as e:
                    logger.warning(f"Brotli decompression failed, trying standard JSON: {e}")
                    data = response.json()
            else:
                data = response.json()
            
            # Cache response
            if self.cache:
                self._cache_json(url, data)
            
            return data
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logger.error(f"Authentication failed. Token may have expired. Status: {e.response.status_code}")
                logger.error("Please extract a fresh token from browser DevTools")
            else:
                logger.error(f"HTTP error for {endpoint}: {e}")
            self.errors += 1
            return None
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout for {endpoint}: {e}")
            self.errors += 1
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {endpoint}: {e}")
            self.errors += 1
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response for {endpoint}: {e}")
            self.errors += 1
            return None
    
    def _get_cached_json(self, url: str) -> Optional[Dict]:
        """Get cached JSON response."""
        cache_path = self.cache._get_cache_path(url).with_suffix('.json')
        if cache_path.exists():
            try:
                import json
                mtime = cache_path.stat().st_mtime
                from datetime import datetime, timedelta
                if datetime.now() - datetime.fromtimestamp(mtime) < self.cache.expiry:
                    return json.loads(cache_path.read_text())
            except Exception:
                pass
        return None
    
    def _cache_json(self, url: str, data: Dict):
        """Cache JSON response."""
        cache_path = self.cache._get_cache_path(url).with_suffix('.json')
        cache_path.write_text(json.dumps(data, indent=2))
    
    # === API Endpoints ===
    
    def get_teams(self, season: int = 2025) -> Optional[List[Dict]]:
        """Get all teams for a season."""
        data = self.fetch_api("teams", params={"season": season})
        if data and isinstance(data, list):
            return data
        elif data and 'teams' in data:
            return data['teams']
        return None
    
    def get_injuries(self, season: int, week: int) -> Optional[List[Dict]]:
        """
        Get injury report for a specific week.
        
        Uses the correct endpoint: /football/v2/injuries
        """
        # Use the correct football API base URL
        url = f"{self.FOOTBALL_API_BASE_URL}/injuries"
        params = {'season': season, 'week': week}
        
        # Use fetch_api but with different base URL
        if url.startswith('http'):
            # Override base URL for this call
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            full_url = f"{url}?{query_string}"
            
            # Check cache
            if self.cache:
                cached = self._get_cached_json(full_url)
                if cached:
                    self.cache_hits += 1
                    return cached.get('injuries', []) if isinstance(cached, dict) else cached
            
            # Rate limit
            self.rate_limiter.wait()
            
            try:
                logger.info(f"Fetching injuries: season={season}, week={week}")
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
                
                # Return injuries list
                if isinstance(data, dict) and 'injuries' in data:
                    return data['injuries']
                elif isinstance(data, list):
                    return data
                
                return []
                
            except Exception as e:
                logger.error(f"Error fetching injuries: {e}")
                self.errors += 1
                return None
        
        return None
    
    def get_player_stats(self, player_id: str, season: int) -> Optional[Dict]:
        """Get player stats for a season."""
        endpoint = f"players/{player_id}/stats"
        data = self.fetch_api(endpoint, params={"season": season})
        return data
    
    def get_game_injuries(self, game_id: str) -> Optional[List[Dict]]:
        """Get injuries for a specific game."""
        endpoint = f"games/{game_id}/injuries"
        data = self.fetch_api(endpoint)
        if data:
            return data if isinstance(data, list) else data.get('injuries', [])
        return None

