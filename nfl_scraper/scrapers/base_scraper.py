"""
Base scraper class with rate limiting, retries, caching, and error handling.
"""

import time
import random
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yaml

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter to avoid overwhelming NFL.com."""
    
    def __init__(self, requests_per_second: float = 0.5):
        self.min_interval = 1.0 / requests_per_second
        self.last_request = 0
    
    def wait(self):
        """Wait if necessary to respect rate limit."""
        elapsed = time.time() - self.last_request
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            # Add small random jitter
            sleep_time += random.uniform(0.1, 0.5)
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_request = time.time()


class ResponseCache:
    """File-based cache for HTTP responses."""
    
    def __init__(self, cache_dir: str, expiry_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.expiry = timedelta(hours=expiry_hours)
    
    def _get_cache_path(self, url: str) -> Path:
        """Get cache file path for a URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"{url_hash}.html"
    
    def get(self, url: str) -> Optional[str]:
        """Get cached response if exists and not expired."""
        cache_path = self._get_cache_path(url)
        
        if not cache_path.exists():
            return None
        
        # Check expiry
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - mtime > self.expiry:
            logger.debug(f"Cache expired for {url}")
            return None
        
        logger.debug(f"Cache hit for {url}")
        return cache_path.read_text(encoding='utf-8')
    
    def set(self, url: str, content: str):
        """Cache response content."""
        cache_path = self._get_cache_path(url)
        cache_path.write_text(content, encoding='utf-8')
        logger.debug(f"Cached response for {url}")


class BaseScraper:
    """
    Base class for NFL.com scrapers.
    
    Features:
    - Rate limiting
    - Automatic retries with exponential backoff
    - Response caching
    - Rotating user agents
    - Comprehensive error handling
    """
    
    def __init__(self, config_path: str = "config/scraping_config.yaml"):
        # Load config
        config_file = Path(__file__).parent.parent / config_path
        if not config_file.exists():
            # Try absolute path
            config_file = Path(config_path)
        with open(config_file) as f:
            self.config = yaml.safe_load(f)
        
        # Resolve cache directory relative to nfl_scraper root
        cache_dir = self.config['cache']['directory']
        if not Path(cache_dir).is_absolute():
            cache_dir = str(Path(__file__).parent.parent / cache_dir)
        
        # Initialize components
        self.rate_limiter = RateLimiter(
            self.config['rate_limit']['requests_per_second']
        )
        self.cache = ResponseCache(
            cache_dir,
            self.config['cache']['expiry_hours']
        ) if self.config['cache']['enabled'] else None
        
        # Setup session with retries
        self.session = self._create_session()
        
        # User agents for rotation
        self.user_agents = self.config['request']['user_agents']
        
        # Stats
        self.requests_made = 0
        self.cache_hits = 0
        self.errors = 0
    
    def _create_session(self) -> requests.Session:
        """Create session with retry configuration."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.config['rate_limit']['max_retries'],
            backoff_factor=self.config['rate_limit']['backoff_multiplier'],
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with rotated user agent."""
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def fetch(self, url: str, use_cache: bool = True) -> Optional[str]:
        """
        Fetch URL content with rate limiting, caching, and error handling.
        
        Args:
            url: URL to fetch
            use_cache: Whether to use cache (default True)
            
        Returns:
            HTML content or None if failed
        """
        # Check cache first
        if use_cache and self.cache:
            cached = self.cache.get(url)
            if cached:
                self.cache_hits += 1
                return cached
        
        # Rate limit
        self.rate_limiter.wait()
        
        try:
            logger.info(f"Fetching: {url}")
            response = self.session.get(
                url,
                headers=self._get_headers(),
                timeout=self.config['request']['timeout']
            )
            response.raise_for_status()
            
            self.requests_made += 1
            content = response.text
            
            # Cache response
            if self.cache:
                self.cache.set(url, content)
            
            return content
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for {url}: {e}")
            self.errors += 1
            return None
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout for {url}: {e}")
            self.errors += 1
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {url}: {e}")
            self.errors += 1
            return None
    
    def get_stats(self) -> Dict[str, int]:
        """Get scraping statistics."""
        return {
            'requests_made': self.requests_made,
            'cache_hits': self.cache_hits,
            'errors': self.errors,
        }

