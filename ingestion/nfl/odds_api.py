"""
The Odds API Client

Client for The Odds API (https://the-odds-api.com/).
Free tier: 500 requests/month

Provides:
- Current NFL odds fetching
- Historical odds (if available)
- Caching to minimize API calls
- Edge calculation utilities
"""

import requests
from datetime import datetime, timedelta
import pandas as pd
from typing import Optional, List, Dict
import json
import os
from pathlib import Path
import logging
import time
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OddsAPIClient:
    """
    Client for The Odds API.
    
    Free tier: 500 requests/month
    Caches responses to minimize API calls.
    """
    
    BASE_URL = "https://api.the-odds-api.com/v4"
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "data/odds_cache"):
        """
        Args:
            api_key: API key (or set ODDS_API_KEY env var or in credentials.yaml)
            cache_dir: Directory for caching responses
        """
        self.api_key = api_key or self._get_api_key()
        if not self.api_key:
            raise ValueError(
                "API key required. Set ODDS_API_KEY env var, pass api_key parameter, "
                "or add to config/credentials.yaml"
            )
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.requests_remaining = None
        self.requests_used = None
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment or config file."""
        # Try environment variable first
        api_key = os.environ.get('ODDS_API_KEY')
        if api_key:
            return api_key
        
        # Try credentials.yaml
        credentials_path = Path(__file__).parent.parent.parent / "config" / "credentials.yaml"
        if credentials_path.exists():
            try:
                with open(credentials_path, 'r') as f:
                    creds = yaml.safe_load(f)
                    if creds and 'odds_api' in creds:
                        return creds['odds_api'].get('api_key')
            except Exception as e:
                logger.warning(f"Error reading credentials.yaml: {e}")
        
        return None
    
    def get_nfl_odds(
        self,
        markets: List[str] = ['spreads', 'h2h', 'totals'],
        bookmakers: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch current NFL odds.
        
        Args:
            markets: List of markets ('spreads', 'h2h', 'totals')
            bookmakers: List of bookmakers (None = all available)
            use_cache: Whether to use cached data if available
        
        Returns:
            DataFrame with columns:
            - game_id, home_team, away_team, commence_time
            - bookmaker, market, outcome_name, price, point
        """
        # Check cache first
        cache_key = f"current_odds_{','.join(sorted(markets))}"
        if use_cache:
            cached = self._get_cached(cache_key, max_age_hours=1)
            if cached is not None:
                logger.info("Using cached odds data")
                return cached
        
        url = f"{self.BASE_URL}/sports/americanfootball_nfl/odds"
        params = {
            'apiKey': self.api_key,
            'markets': ','.join(markets),
            'regions': 'us',
        }
        
        if bookmakers:
            params['bookmakers'] = ','.join(bookmakers)
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            # Extract rate limit info from headers
            self.requests_remaining = response.headers.get('x-requests-remaining')
            self.requests_used = response.headers.get('x-requests-used')
            
            if self.requests_remaining:
                remaining = int(self.requests_remaining)
                logger.info(f"API requests remaining: {remaining}")
                if remaining < 50:
                    logger.warning(f"Low API quota: {remaining} requests remaining")
            
            data = response.json()
            
            # Parse response into DataFrame
            odds_list = []
            for game in data:
                game_id = self._extract_game_id(game)
                commence_time = game.get('commence_time', '')
                home_team = self._extract_team(game.get('home_team', ''))
                away_team = self._extract_team(game.get('away_team', ''))
                
                # Extract odds from bookmakers
                for bookmaker in game.get('bookmakers', []):
                    bookmaker_key = bookmaker.get('key', '')
                    
                    for market in bookmaker.get('markets', []):
                        market_key = market.get('key', '')
                        
                        for outcome in market.get('outcomes', []):
                            odds_list.append({
                                'game_id': game_id,
                                'home_team': home_team,
                                'away_team': away_team,
                                'commence_time': commence_time,
                                'bookmaker': bookmaker_key,
                                'market': market_key,
                                'outcome_name': outcome.get('name', ''),
                                'price': outcome.get('price', None),
                                'point': outcome.get('point', None),
                            })
            
            df = pd.DataFrame(odds_list)
            
            # Cache the result
            if use_cache:
                self._cache_response(cache_key, df)
            
            logger.info(f"Fetched {len(df)} odds entries for {len(data)} games")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching odds from API: {e}")
            raise
    
    def get_historical_odds(
        self,
        date: str,
        markets: List[str] = ['spreads'],
    ) -> pd.DataFrame:
        """
        Fetch historical odds (requires paid plan).
        For backtesting, we may need to use cached/CSV data instead.
        
        Args:
            date: Date string (YYYY-MM-DD)
            markets: List of markets
        
        Returns:
            DataFrame with historical odds
        """
        logger.warning("Historical odds endpoint may require paid plan")
        
        url = f"{self.BASE_URL}/sports/americanfootball_nfl/odds"
        params = {
            'apiKey': self.api_key,
            'markets': ','.join(markets),
            'regions': 'us',
            'dateFormat': 'iso',
        }
        
        # The Odds API v4 doesn't have a direct historical endpoint
        # This would need to be implemented differently or use cached data
        logger.warning("Historical odds not directly supported by free API")
        return pd.DataFrame()
    
    def get_best_odds(
        self,
        odds_df: pd.DataFrame,
        team: str,
        market: str = 'spreads',
    ) -> Dict:
        """
        Find best available odds for a team across all bookmakers.
        
        Args:
            odds_df: DataFrame from get_nfl_odds()
            team: Team abbreviation
            market: Market type ('spreads', 'h2h', 'totals')
        
        Returns:
            {
                'bookmaker': 'fanduel',
                'price': -108,
                'point': -3.5,  # For spreads
            }
        """
        if len(odds_df) == 0:
            return {}
        
        # Filter to market and team
        market_odds = odds_df[
            (odds_df['market'] == market) &
            ((odds_df['outcome_name'].str.contains(team, case=False)) |
             (odds_df['home_team'] == team) |
             (odds_df['away_team'] == team))
        ].copy()
        
        if len(market_odds) == 0:
            return {}
        
        # For spreads, best odds = highest price (most favorable)
        # For totals, depends on over/under
        if market == 'spreads':
            # Find best price (highest for favorite, lowest for underdog)
            best_idx = market_odds['price'].idxmax()
        else:
            # For other markets, use highest price
            best_idx = market_odds['price'].idxmax()
        
        best = market_odds.loc[best_idx]
        
        return {
            'bookmaker': best['bookmaker'],
            'price': best['price'],
            'point': best.get('point'),
        }
    
    def calculate_implied_probability(self, american_odds: int) -> float:
        """
        Convert American odds to implied probability.
        
        Args:
            american_odds: American odds (e.g., -110, +150)
        
        Returns:
            Implied probability (0.0 to 1.0)
        """
        if american_odds < 0:
            return abs(american_odds) / (abs(american_odds) + 100)
        else:
            return 100 / (american_odds + 100)
    
    def calculate_edge(self, model_prob: float, market_prob: float) -> float:
        """
        Calculate betting edge.
        
        Edge = Model Probability - Market Implied Probability
        Positive edge = model sees value the market doesn't
        
        Args:
            model_prob: Model's predicted probability (0.0 to 1.0)
            market_prob: Market's implied probability (0.0 to 1.0)
        
        Returns:
            Edge value (positive = value bet)
        """
        return model_prob - market_prob
    
    def _cache_response(self, cache_key: str, data: pd.DataFrame):
        """Cache API response to reduce API calls."""
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        try:
            data.to_parquet(cache_file)
            logger.debug(f"Cached response to {cache_file}")
        except Exception as e:
            logger.warning(f"Error caching response: {e}")
    
    def _get_cached(self, cache_key: str, max_age_hours: float = 1.0) -> Optional[pd.DataFrame]:
        """
        Retrieve cached response if fresh.
        
        Args:
            cache_key: Cache key
            max_age_hours: Maximum age of cache in hours
        
        Returns:
            Cached DataFrame or None if not available/fresh
        """
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        
        if not cache_file.exists():
            return None
        
        # Check age
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if file_age > timedelta(hours=max_age_hours):
            logger.debug(f"Cache expired (age: {file_age})")
            return None
        
        try:
            df = pd.read_parquet(cache_file)
            logger.debug(f"Loaded cached data from {cache_file}")
            return df
        except Exception as e:
            logger.warning(f"Error loading cache: {e}")
            return None
    
    def _extract_game_id(self, game: dict) -> str:
        """Extract game_id from API response."""
        # Try to construct game_id from teams and date
        home_team = self._extract_team(game.get('home_team', ''))
        away_team = self._extract_team(game.get('away_team', ''))
        commence_time = game.get('commence_time', '')
        
        try:
            game_date = pd.to_datetime(commence_time)
            season = game_date.year if game_date.month >= 9 else game_date.year - 1
            # Week would need to be calculated from date
            week = 1  # Placeholder
            return f"nfl_{season}_{week:02d}_{away_team}_{home_team}"
        except:
            return f"nfl_{away_team}_{home_team}_{commence_time}"
    
    def _extract_team(self, team_name: str) -> str:
        """Extract team abbreviation from full name."""
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


# CLI entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch NFL odds from The Odds API")
    parser.add_argument('--fetch-current', action='store_true', help='Fetch current week odds')
    parser.add_argument('--markets', nargs='+', default=['spreads'], help='Markets to fetch')
    parser.add_argument('--output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    if args.fetch_current:
        client = OddsAPIClient()
        odds_df = client.get_nfl_odds(markets=args.markets)
        
        if args.output:
            odds_df.to_parquet(args.output)
            print(f"Saved odds to {args.output}")
        else:
            print(odds_df.to_string())

