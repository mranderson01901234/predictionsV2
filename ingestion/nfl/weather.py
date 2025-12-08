"""
Weather Data Ingestion for NFL Games

Uses Open-Meteo API (free, no API key required):
- Historical weather: https://archive-api.open-meteo.com/v1/archive
- Forecast weather: https://api.open-meteo.com/v1/forecast

Weather primarily affects outdoor games (~50% of games).
Dome/indoor stadiums have controlled conditions.
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from pathlib import Path
import logging
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# All 32 NFL team stadiums with coordinates and dome status
STADIUMS = {
    # AFC East
    'BUF': {'name': 'Highmark Stadium', 'lat': 42.7738, 'lon': -78.7870, 'dome': False, 'timezone': 'America/New_York'},
    'MIA': {'name': 'Hard Rock Stadium', 'lat': 25.9580, 'lon': -80.2389, 'dome': False, 'timezone': 'America/New_York'},
    'NE': {'name': 'Gillette Stadium', 'lat': 42.0909, 'lon': -71.2643, 'dome': False, 'timezone': 'America/New_York'},
    'NYJ': {'name': 'MetLife Stadium', 'lat': 40.8135, 'lon': -74.0745, 'dome': False, 'timezone': 'America/New_York'},
    
    # AFC North
    'BAL': {'name': 'M&T Bank Stadium', 'lat': 39.2780, 'lon': -76.6227, 'dome': False, 'timezone': 'America/New_York'},
    'CIN': {'name': 'Paycor Stadium', 'lat': 39.0955, 'lon': -84.5161, 'dome': False, 'timezone': 'America/New_York'},
    'CLE': {'name': 'Cleveland Browns Stadium', 'lat': 41.5061, 'lon': -81.6995, 'dome': False, 'timezone': 'America/New_York'},
    'PIT': {'name': 'Acrisure Stadium', 'lat': 40.4468, 'lon': -80.0158, 'dome': False, 'timezone': 'America/New_York'},
    
    # AFC South
    'HOU': {'name': 'NRG Stadium', 'lat': 29.6847, 'lon': -95.4107, 'dome': True, 'timezone': 'America/Chicago'},
    'IND': {'name': 'Lucas Oil Stadium', 'lat': 39.7601, 'lon': -86.1639, 'dome': True, 'timezone': 'America/Indiana/Indianapolis'},
    'JAX': {'name': 'EverBank Stadium', 'lat': 30.3239, 'lon': -81.6373, 'dome': False, 'timezone': 'America/New_York'},
    'TEN': {'name': 'Nissan Stadium', 'lat': 36.1665, 'lon': -86.7713, 'dome': False, 'timezone': 'America/Chicago'},
    
    # AFC West
    'DEN': {'name': 'Empower Field', 'lat': 39.7439, 'lon': -105.0201, 'dome': False, 'timezone': 'America/Denver'},
    'KC': {'name': 'GEHA Field', 'lat': 39.0489, 'lon': -94.4839, 'dome': False, 'timezone': 'America/Chicago'},
    'LV': {'name': 'Allegiant Stadium', 'lat': 36.0909, 'lon': -115.1833, 'dome': True, 'timezone': 'America/Los_Angeles'},
    'LAC': {'name': 'SoFi Stadium', 'lat': 33.9535, 'lon': -118.3392, 'dome': True, 'timezone': 'America/Los_Angeles'},
    
    # NFC East
    'DAL': {'name': 'AT&T Stadium', 'lat': 32.7473, 'lon': -97.0945, 'dome': True, 'timezone': 'America/Chicago'},
    'NYG': {'name': 'MetLife Stadium', 'lat': 40.8135, 'lon': -74.0745, 'dome': False, 'timezone': 'America/New_York'},
    'PHI': {'name': 'Lincoln Financial Field', 'lat': 39.9008, 'lon': -75.1675, 'dome': False, 'timezone': 'America/New_York'},
    'WAS': {'name': 'Commanders Field', 'lat': 38.9076, 'lon': -76.8645, 'dome': False, 'timezone': 'America/New_York'},
    
    # NFC North
    'CHI': {'name': 'Soldier Field', 'lat': 41.8623, 'lon': -87.6167, 'dome': False, 'timezone': 'America/Chicago'},
    'DET': {'name': 'Ford Field', 'lat': 42.3400, 'lon': -83.0456, 'dome': True, 'timezone': 'America/Detroit'},
    'GB': {'name': 'Lambeau Field', 'lat': 44.5013, 'lon': -88.0622, 'dome': False, 'timezone': 'America/Chicago'},
    'MIN': {'name': 'U.S. Bank Stadium', 'lat': 44.9737, 'lon': -93.2577, 'dome': True, 'timezone': 'America/Chicago'},
    
    # NFC South
    'ATL': {'name': 'Mercedes-Benz Stadium', 'lat': 33.7554, 'lon': -84.4010, 'dome': True, 'timezone': 'America/New_York'},
    'CAR': {'name': 'Bank of America Stadium', 'lat': 35.2258, 'lon': -80.8528, 'dome': False, 'timezone': 'America/New_York'},
    'NO': {'name': 'Caesars Superdome', 'lat': 29.9511, 'lon': -90.0812, 'dome': True, 'timezone': 'America/Chicago'},
    'TB': {'name': 'Raymond James Stadium', 'lat': 27.9759, 'lon': -82.5033, 'dome': False, 'timezone': 'America/New_York'},
    
    # NFC West
    'ARI': {'name': 'State Farm Stadium', 'lat': 33.5276, 'lon': -112.2626, 'dome': True, 'timezone': 'America/Phoenix'},
    'LAR': {'name': 'SoFi Stadium', 'lat': 33.9535, 'lon': -118.3392, 'dome': True, 'timezone': 'America/Los_Angeles'},
    'SF': {'name': "Levi's Stadium", 'lat': 37.4033, 'lon': -121.9694, 'dome': False, 'timezone': 'America/Los_Angeles'},
    'SEA': {'name': 'Lumen Field', 'lat': 47.5952, 'lon': -122.3316, 'dome': False, 'timezone': 'America/Los_Angeles'},
}

# Teams with indoor/dome stadiums (weather doesn't affect game)
DOME_TEAMS = [team for team, info in STADIUMS.items() if info['dome']]
# ['HOU', 'IND', 'LV', 'LAC', 'DAL', 'DET', 'MIN', 'ATL', 'NO', 'ARI', 'LAR']


class WeatherIngestion:
    """
    Fetch weather data for NFL games.
    
    Uses Open-Meteo API:
    - Free, no API key required
    - Historical data available back to 1940
    - Hourly resolution
    """
    
    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    
    def __init__(self, cache_dir: str = "data/nfl/cache/weather"):
        """
        Args:
            cache_dir: Directory for caching weather responses
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_game_weather(
        self,
        home_team: str,
        game_datetime: datetime,
        use_cache: bool = True,
    ) -> Dict:
        """
        Get weather conditions for a game.
        
        Args:
            home_team: Home team abbreviation
            game_datetime: Game start time (datetime object)
            use_cache: Whether to use cached data if available
        
        Returns:
            {
                'temperature_f': 45,
                'feels_like_f': 38,
                'wind_speed_mph': 12,
                'wind_gust_mph': 22,
                'wind_direction': 'NW',
                'precipitation_prob': 20,
                'precipitation_inches': 0.0,
                'humidity_pct': 65,
                'visibility_miles': 10,
                'weather_code': 3,  # WMO code
                'weather_description': 'Overcast',
                'is_dome': False,
            }
        """
        # Check if dome stadium
        stadium = STADIUMS.get(home_team)
        if not stadium:
            logger.warning(f"Unknown team: {home_team}, using default weather")
            return self._get_dome_conditions()
        
        if stadium['dome']:
            return self._get_dome_conditions()
        
        # Check cache first
        if use_cache:
            cached = self._get_cached(home_team, game_datetime)
            if cached is not None:
                return cached
        
        # Fetch from Open-Meteo
        if game_datetime < datetime.now():
            weather_data = self._fetch_historical(stadium, game_datetime)
        else:
            weather_data = self._fetch_forecast(stadium, game_datetime)
        
        # Cache the result
        if use_cache and weather_data:
            self._cache_response(home_team, game_datetime, weather_data)
        
        return weather_data
    
    def _fetch_historical(self, stadium: Dict, game_datetime: datetime) -> Dict:
        """
        Fetch historical weather from Open-Meteo archive.
        
        Endpoint: https://archive-api.open-meteo.com/v1/archive
        """
        url = self.ARCHIVE_URL
        
        # Get game date
        game_date = game_datetime.date()
        
        # Estimate game hour (most NFL games start 1pm, 4pm, or 8pm local time)
        game_hour = game_datetime.hour
        
        params = {
            'latitude': stadium['lat'],
            'longitude': stadium['lon'],
            'start_date': game_date.strftime('%Y-%m-%d'),
            'end_date': game_date.strftime('%Y-%m-%d'),
            'hourly': 'temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_gusts_10m,wind_direction_10m,weather_code',
            'temperature_unit': 'fahrenheit',
            'wind_speed_unit': 'mph',
            'precipitation_unit': 'inch',
            'timezone': stadium['timezone'],
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Extract hourly data
            hourly = data.get('hourly', {})
            if not hourly or 'time' not in hourly:
                logger.warning(f"No hourly data for {game_date}")
                return self._get_default_weather()
            
            # Find closest hour to game time
            times = hourly['time']
            temperatures = hourly.get('temperature_2m', [])
            humidity = hourly.get('relative_humidity_2m', [])
            precipitation = hourly.get('precipitation', [])
            wind_speed = hourly.get('wind_speed_10m', [])
            wind_gusts = hourly.get('wind_gusts_10m', [])
            wind_direction = hourly.get('wind_direction_10m', [])
            weather_codes = hourly.get('weather_code', [])
            
            # Find index closest to game hour
            if len(times) == 0:
                return self._get_default_weather()
            
            # Use game hour or closest available
            idx = min(game_hour, len(times) - 1) if game_hour < len(times) else len(times) - 1
            
            # Extract values
            temp_f = temperatures[idx] if idx < len(temperatures) else 60.0
            humidity_pct = humidity[idx] if idx < len(humidity) else 50.0
            precip_inches = precipitation[idx] if idx < len(precipitation) else 0.0
            wind_mph = wind_speed[idx] if idx < len(wind_speed) else 0.0
            wind_gust_mph = wind_gusts[idx] if idx < len(wind_gusts) else wind_mph
            wind_dir_deg = wind_direction[idx] if idx < len(wind_direction) else 0
            weather_code = weather_codes[idx] if idx < len(weather_codes) else 0
            
            # Convert wind direction to cardinal
            wind_direction_cardinal = self._degrees_to_cardinal(wind_dir_deg)
            
            # Get weather description from code
            weather_description = self._weather_code_to_description(weather_code)
            
            # Calculate feels like (wind chill for cold, heat index for hot)
            feels_like_f = self._calculate_feels_like(temp_f, wind_mph, humidity_pct)
            
            # Precipitation probability (simplified - use actual if available, else estimate)
            precip_prob = 0.0
            if precip_inches > 0:
                precip_prob = min(100, precip_inches * 50)  # Rough estimate
            
            return {
                'temperature_f': round(temp_f, 1),
                'feels_like_f': round(feels_like_f, 1),
                'wind_speed_mph': round(wind_mph, 1),
                'wind_gust_mph': round(wind_gust_mph, 1),
                'wind_direction': wind_direction_cardinal,
                'precipitation_prob': round(precip_prob, 1),
                'precipitation_inches': round(precip_inches, 2),
                'humidity_pct': round(humidity_pct, 1),
                'visibility_miles': 10.0,  # Default, API doesn't always provide
                'weather_code': int(weather_code),
                'weather_description': weather_description,
                'is_dome': False,
            }
            
        except Exception as e:
            logger.error(f"Error fetching historical weather: {e}")
            return self._get_default_weather()
    
    def _fetch_forecast(self, stadium: Dict, game_datetime: datetime) -> Dict:
        """
        Fetch forecast weather from Open-Meteo.
        
        Endpoint: https://api.open-meteo.com/v1/forecast
        """
        url = self.FORECAST_URL
        
        params = {
            'latitude': stadium['lat'],
            'longitude': stadium['lon'],
            'hourly': 'temperature_2m,relative_humidity_2m,precipitation_probability,precipitation,wind_speed_10m,wind_gusts_10m,wind_direction_10m,weather_code',
            'temperature_unit': 'fahrenheit',
            'wind_speed_unit': 'mph',
            'precipitation_unit': 'inch',
            'timezone': stadium['timezone'],
            'forecast_days': 7,  # Get 7-day forecast
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            hourly = data.get('hourly', {})
            if not hourly or 'time' not in hourly:
                logger.warning(f"No forecast data available")
                return self._get_default_weather()
            
            # Find closest hour to game time
            times = hourly['time']
            game_time_str = game_datetime.strftime('%Y-%m-%dT%H:00')
            
            # Find matching or closest time
            idx = 0
            for i, t in enumerate(times):
                if t.startswith(game_time_str[:13]):  # Match date and hour
                    idx = i
                    break
                if t > game_time_str:
                    idx = max(0, i - 1)
                    break
            
            if idx >= len(times):
                idx = len(times) - 1
            
            # Extract values
            temperatures = hourly.get('temperature_2m', [])
            humidity = hourly.get('relative_humidity_2m', [])
            precip_prob = hourly.get('precipitation_probability', [])
            precipitation = hourly.get('precipitation', [])
            wind_speed = hourly.get('wind_speed_10m', [])
            wind_gusts = hourly.get('wind_gusts_10m', [])
            wind_direction = hourly.get('wind_direction_10m', [])
            weather_codes = hourly.get('weather_code', [])
            
            temp_f = temperatures[idx] if idx < len(temperatures) else 60.0
            humidity_pct = humidity[idx] if idx < len(humidity) else 50.0
            precip_prob_pct = precip_prob[idx] if idx < len(precip_prob) else 0.0
            precip_inches = precipitation[idx] if idx < len(precipitation) else 0.0
            wind_mph = wind_speed[idx] if idx < len(wind_speed) else 0.0
            wind_gust_mph = wind_gusts[idx] if idx < len(wind_gusts) else wind_mph
            wind_dir_deg = wind_direction[idx] if idx < len(wind_direction) else 0
            weather_code = weather_codes[idx] if idx < len(weather_codes) else 0
            
            wind_direction_cardinal = self._degrees_to_cardinal(wind_dir_deg)
            weather_description = self._weather_code_to_description(weather_code)
            feels_like_f = self._calculate_feels_like(temp_f, wind_mph, humidity_pct)
            
            return {
                'temperature_f': round(temp_f, 1),
                'feels_like_f': round(feels_like_f, 1),
                'wind_speed_mph': round(wind_mph, 1),
                'wind_gust_mph': round(wind_gust_mph, 1),
                'wind_direction': wind_direction_cardinal,
                'precipitation_prob': round(precip_prob_pct, 1),
                'precipitation_inches': round(precip_inches, 2),
                'humidity_pct': round(humidity_pct, 1),
                'visibility_miles': 10.0,
                'weather_code': int(weather_code),
                'weather_description': weather_description,
                'is_dome': False,
            }
            
        except Exception as e:
            logger.error(f"Error fetching forecast weather: {e}")
            return self._get_default_weather()
    
    def _get_dome_conditions(self) -> Dict:
        """Return standard dome conditions (controlled environment)."""
        return {
            'temperature_f': 72.0,
            'feels_like_f': 72.0,
            'wind_speed_mph': 0.0,
            'wind_gust_mph': 0.0,
            'wind_direction': 'N',
            'precipitation_prob': 0.0,
            'precipitation_inches': 0.0,
            'humidity_pct': 50.0,
            'visibility_miles': 10.0,
            'weather_code': 0,
            'weather_description': 'Indoor/Dome',
            'is_dome': True,
        }
    
    def _get_default_weather(self) -> Dict:
        """Return default weather when API fails."""
        return {
            'temperature_f': 60.0,
            'feels_like_f': 60.0,
            'wind_speed_mph': 5.0,
            'wind_gust_mph': 8.0,
            'wind_direction': 'SW',
            'precipitation_prob': 0.0,
            'precipitation_inches': 0.0,
            'humidity_pct': 50.0,
            'visibility_miles': 10.0,
            'weather_code': 0,
            'weather_description': 'Clear',
            'is_dome': False,
        }
    
    def fetch_season_weather(self, season: int, games_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Fetch weather for all games in a season (for backtesting).
        
        Args:
            season: Season year
            games_df: Optional DataFrame with games (must have: home_team, gameday or date)
        
        Returns:
            DataFrame with weather data for each game
        """
        logger.info(f"Fetching weather for season {season}...")
        
        if games_df is None:
            logger.warning("No games DataFrame provided, cannot fetch season weather")
            return pd.DataFrame()
        
        weather_records = []
        
        for idx, game in games_df.iterrows():
            if game.get('season') != season:
                continue
            
            home_team = game.get('home_team')
            game_date = game.get('gameday') or game.get('date')
            
            if pd.isna(game_date):
                continue
            
            # Convert to datetime if needed
            if isinstance(game_date, str):
                try:
                    game_datetime = pd.to_datetime(game_date)
                except:
                    continue
            else:
                game_datetime = game_date
            
            # Fetch weather
            weather = self.get_game_weather(home_team, game_datetime, use_cache=True)
            
            weather_records.append({
                'game_id': game.get('game_id'),
                'season': season,
                'week': game.get('week'),
                'home_team': home_team,
                **weather
            })
            
            # Rate limiting
            time.sleep(0.1)  # Small delay to avoid overwhelming API
        
        df = pd.DataFrame(weather_records)
        logger.info(f"Fetched weather for {len(df)} games")
        
        return df
    
    def _degrees_to_cardinal(self, degrees: float) -> str:
        """Convert wind direction degrees to cardinal direction."""
        directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                     'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        idx = int((degrees + 11.25) / 22.5) % 16
        return directions[idx]
    
    def _weather_code_to_description(self, code: int) -> str:
        """Convert WMO weather code to description."""
        code_map = {
            0: 'Clear',
            1: 'Mostly Clear',
            2: 'Partly Cloudy',
            3: 'Overcast',
            45: 'Foggy',
            48: 'Depositing Rime Fog',
            51: 'Light Drizzle',
            53: 'Moderate Drizzle',
            55: 'Dense Drizzle',
            56: 'Light Freezing Drizzle',
            57: 'Dense Freezing Drizzle',
            61: 'Slight Rain',
            63: 'Moderate Rain',
            65: 'Heavy Rain',
            66: 'Light Freezing Rain',
            67: 'Heavy Freezing Rain',
            71: 'Slight Snow',
            73: 'Moderate Snow',
            75: 'Heavy Snow',
            77: 'Snow Grains',
            80: 'Slight Rain Showers',
            81: 'Moderate Rain Showers',
            82: 'Violent Rain Showers',
            85: 'Slight Snow Showers',
            86: 'Heavy Snow Showers',
            95: 'Thunderstorm',
            96: 'Thunderstorm with Hail',
            99: 'Thunderstorm with Heavy Hail',
        }
        return code_map.get(code, 'Unknown')
    
    def _calculate_feels_like(self, temp_f: float, wind_mph: float, humidity_pct: float) -> float:
        """
        Calculate feels-like temperature (wind chill for cold, heat index for hot).
        
        Simplified calculation - for more accuracy, use full formulas.
        """
        if temp_f <= 50 and wind_mph > 3:
            # Wind chill formula (simplified)
            feels_like = 35.74 + 0.6215 * temp_f - 35.75 * (wind_mph ** 0.16) + 0.4275 * temp_f * (wind_mph ** 0.16)
        elif temp_f >= 80:
            # Heat index formula (simplified)
            feels_like = -42.379 + 2.04901523 * temp_f + 10.14333127 * humidity_pct - 0.22475541 * temp_f * humidity_pct
        else:
            feels_like = temp_f
        
        return feels_like
    
    def _cache_response(self, team: str, game_datetime: datetime, weather_data: Dict):
        """Cache weather response to reduce API calls."""
        cache_key = f"{team}_{game_datetime.strftime('%Y%m%d')}"
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(weather_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Error caching weather: {e}")
    
    def _get_cached(self, team: str, game_datetime: datetime, max_age_hours: float = 24.0) -> Optional[Dict]:
        """
        Retrieve cached weather response if fresh.
        
        Args:
            team: Team abbreviation
            game_datetime: Game datetime
            max_age_hours: Maximum age of cache in hours
        
        Returns:
            Cached weather dict or None if not available/fresh
        """
        cache_key = f"{team}_{game_datetime.strftime('%Y%m%d')}"
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        # Check age
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if file_age > timedelta(hours=max_age_hours):
            return None
        
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading cache: {e}")
            return None


# Convenience function
def get_game_weather(home_team: str, game_datetime: datetime) -> Dict:
    """
    Convenience function to get weather for a game.
    
    Args:
        home_team: Home team abbreviation
        game_datetime: Game start datetime
    
    Returns:
        Weather data dictionary
    """
    ingester = WeatherIngestion()
    return ingester.get_game_weather(home_team, game_datetime)

