"""
Base scraper for FootballDB.

FootballDB serves HTML pages (not JSON APIs), so we need to:
1. Fetch HTML
2. Parse tables with BeautifulSoup
3. Handle rate limiting (be respectful)
"""
import time
import random
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RateLimiter:
    """Rate limiter with jitter."""
    
    min_interval: float = 0.3  # Minimum seconds between requests
    max_interval: float = 0.7  # Maximum seconds (with jitter)
    last_request: float = 0
    
    def wait(self):
        """Wait with random jitter."""
        elapsed = time.time() - self.last_request
        target_interval = random.uniform(self.min_interval, self.max_interval)
        
        if elapsed < target_interval:
            sleep_time = target_interval - elapsed
            time.sleep(sleep_time)
        
        self.last_request = time.time()


class HTMLCache:
    """File-based HTML cache."""
    
    def __init__(
        self, 
        cache_dir: str = "data/cache/footballdb",
        expiry_hours: int = 24
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.expiry = timedelta(hours=expiry_hours)
    
    def _get_cache_key(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()
    
    def _get_cache_path(self, url: str) -> Path:
        return self.cache_dir / f"{self._get_cache_key(url)}.html"
    
    def get(self, url: str) -> Optional[str]:
        """Get cached HTML if exists and not expired."""
        cache_path = self._get_cache_path(url)
        
        if not cache_path.exists():
            return None
        
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - mtime > self.expiry:
            return None
        
        return cache_path.read_text(encoding='utf-8')
    
    def set(self, url: str, html: str):
        """Cache HTML content."""
        cache_path = self._get_cache_path(url)
        cache_path.write_text(html, encoding='utf-8')


class FootballDBScraper:
    """
    Base scraper for FootballDB.
    
    Handles:
    - HTTP requests with proper headers
    - Rate limiting (be nice to their servers)
    - Response caching
    - Error handling
    """
    
    BASE_URL = "https://www.footballdb.com"
    
    # Browser-like headers
    DEFAULT_HEADERS = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }
    
    def __init__(
        self,
        cache_enabled: bool = True,
        cache_dir: str = "data/cache/footballdb"
    ):
        self.rate_limiter = RateLimiter(min_interval=0.3, max_interval=0.7)
        self.cache = HTMLCache(cache_dir) if cache_enabled else None
        
        # Setup session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        
        # Stats
        self.requests_made = 0
        self.cache_hits = 0
    
    def fetch(self, url: str, use_cache: bool = True) -> Optional[str]:
        """
        Fetch HTML from URL.
        
        Args:
            url: Full URL to fetch
            use_cache: Whether to use cache
            
        Returns:
            HTML content as string, or None if failed
        """
        # Check cache
        if use_cache and self.cache:
            cached = self.cache.get(url)
            if cached:
                self.cache_hits += 1
                logger.debug(f"Cache hit: {url}")
                return cached
        
        # Rate limit
        self.rate_limiter.wait()
        
        try:
            logger.info(f"Fetching: {url}")
            
            response = self.session.get(
                url,
                headers=self.DEFAULT_HEADERS,
                timeout=30
            )
            
            self.requests_made += 1
            
            if response.status_code == 200:
                html = response.text
                
                # Cache
                if self.cache:
                    self.cache.set(url, html)
                
                return html
            else:
                logger.error(f"HTTP {response.status_code}: {url}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return None
    
    def fetch_and_parse(self, url: str, use_cache: bool = True) -> Optional[BeautifulSoup]:
        """Fetch and return parsed BeautifulSoup object."""
        html = self.fetch(url, use_cache)
        if html:
            return BeautifulSoup(html, 'html.parser')
        return None
    
    def get_stats(self) -> Dict[str, int]:
        """Get scraper statistics."""
        return {
            'requests_made': self.requests_made,
            'cache_hits': self.cache_hits,
        }

