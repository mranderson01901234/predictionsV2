"""
Parallel Weather Data Ingestion

Optimized version with parallel API calls for faster feature generation.
Uses ThreadPoolExecutor to fetch multiple weather records simultaneously.
"""

import pandas as pd
import requests
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from ingestion.nfl.weather import WeatherIngestion, STADIUMS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParallelWeatherIngestion(WeatherIngestion):
    """
    Parallel version of WeatherIngestion for faster batch processing.
    
    Uses ThreadPoolExecutor to fetch multiple weather records concurrently.
    """
    
    def __init__(self, cache_dir: str = "data/nfl/cache/weather", max_workers: int = 10):
        """
        Args:
            cache_dir: Directory for caching weather responses
            max_workers: Number of parallel threads for API calls
        """
        super().__init__(cache_dir)
        self.max_workers = max_workers
    
    def fetch_season_weather_parallel(
        self,
        season: int,
        games_df: Optional[pd.DataFrame] = None,
        max_workers: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch weather for all games in a season using parallel API calls.
        
        Args:
            season: Season year
            games_df: DataFrame with games (must have: home_team, gameday or date)
            max_workers: Number of parallel threads (default: self.max_workers)
        
        Returns:
            DataFrame with weather data for each game
        """
        logger.info(f"Fetching weather for season {season} using {max_workers or self.max_workers} parallel workers...")
        
        if games_df is None:
            logger.warning("No games DataFrame provided, cannot fetch season weather")
            return pd.DataFrame()
        
        # Filter to season
        season_games = games_df[games_df['season'] == season].copy()
        
        if len(season_games) == 0:
            logger.warning(f"No games found for season {season}")
            return pd.DataFrame()
        
        logger.info(f"Processing {len(season_games)} games in parallel...")
        
        # Prepare arguments for parallel processing
        weather_records = []
        workers = max_workers or self.max_workers
        
        # Use ThreadPoolExecutor for parallel API calls
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_game = {}
            for idx, game in season_games.iterrows():
                home_team = game.get('home_team')
                game_date = game.get('gameday') or game.get('date')
                
                if pd.isna(game_date) or not home_team:
                    continue
                
                # Convert to datetime if needed
                if isinstance(game_date, str):
                    try:
                        game_datetime = pd.to_datetime(game_date)
                    except:
                        continue
                else:
                    game_datetime = game_date
                
                # Check cache first (fast, can do synchronously)
                cached = self._get_cached(home_team, game_datetime)
                if cached is not None:
                    weather_records.append({
                        'game_id': game.get('game_id'),
                        'season': season,
                        'week': game.get('week'),
                        'home_team': home_team,
                        **cached
                    })
                    continue
                
                # Submit API call task
                future = executor.submit(
                    self._fetch_weather_for_game,
                    home_team,
                    game_datetime,
                    game.get('game_id'),
                    season,
                    game.get('week'),
                )
                future_to_game[future] = (home_team, game_datetime)
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_game):
                try:
                    result = future.result()
                    if result:
                        weather_records.append(result)
                    completed += 1
                    
                    if completed % 50 == 0:
                        logger.info(f"  Progress: {completed}/{len(future_to_game)} weather records fetched")
                except Exception as e:
                    home_team, game_datetime = future_to_game[future]
                    logger.warning(f"Error fetching weather for {home_team} {game_datetime}: {e}")
                    completed += 1
        
        df = pd.DataFrame(weather_records)
        logger.info(f"Fetched weather for {len(df)} games using parallel processing")
        
        return df
    
    def _fetch_weather_for_game(
        self,
        home_team: str,
        game_datetime: datetime,
        game_id: Optional[str],
        season: Optional[int],
        week: Optional[int],
    ) -> Optional[Dict]:
        """
        Fetch weather for a single game (for parallel processing).
        
        Returns:
            Dictionary with weather data and game metadata, or None if error
        """
        try:
            weather = self.get_game_weather(home_team, game_datetime, use_cache=True)
            
            if weather:
                return {
                    'game_id': game_id,
                    'season': season,
                    'week': week,
                    'home_team': home_team,
                    **weather
                }
        except Exception as e:
            logger.warning(f"Error in parallel fetch for {home_team} {game_datetime}: {e}")
        
        return None


def fetch_weather_parallel_batch(
    games_df: pd.DataFrame,
    max_workers: int = 10,
    batch_size: int = 100,
) -> pd.DataFrame:
    """
    Fetch weather for all games in a DataFrame using parallel processing.
    
    Args:
        games_df: DataFrame with games
        max_workers: Number of parallel threads
        batch_size: Process in batches to avoid memory issues
    
    Returns:
        DataFrame with weather data
    """
    ingester = ParallelWeatherIngestion(max_workers=max_workers)
    
    all_weather = []
    seasons = sorted(games_df['season'].unique())
    
    for season in seasons:
        logger.info(f"Processing season {season}...")
        season_weather = ingester.fetch_season_weather_parallel(
            season=season,
            games_df=games_df,
            max_workers=max_workers,
        )
        if len(season_weather) > 0:
            all_weather.append(season_weather)
    
    if len(all_weather) == 0:
        return pd.DataFrame()
    
    df = pd.concat(all_weather, ignore_index=True)
    logger.info(f"Total weather records fetched: {len(df)}")
    
    return df

