"""
Scraper for NGS Game Center endpoint.

Game Center provides per-game detailed stats including:
- All passers in the game with NGS metrics
- All rushers with RYOE
- All receivers with separation/YAC
- Top plays by each team
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd

from .base_client import NGSClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GameCenterScraper:
    """
    Scrape per-game statistics from NGS Game Center.
    
    Provides granular game-level data for each player.
    """
    
    def __init__(self, output_dir: str = "data/raw/gamecenter"):
        self.client = NGSClient()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_game_ids_for_season(self, season: int) -> List[str]:
        """
        Get all game IDs for a season.
        
        Game IDs follow format: YYYYMMDDHH
        Need to derive from schedule data.
        """
        # Import schedule from nfl_data_py
        try:
            import nfl_data_py as nfl
            schedule = nfl.import_schedules([season])
            
            # Filter completed games
            schedule = schedule[schedule['result'].notna()]
            
            # Extract game IDs
            game_ids = schedule['game_id'].tolist()
            
            # Convert to NGS format if needed
            # NFL game_id: 2024_01_KC_BAL
            # NGS game_id: 2024090508 (YYYYMMDDHH)
            
            # We need to map these - for now, use gameday column
            ngs_ids = []
            for _, row in schedule.iterrows():
                if pd.notna(row.get('gameday')):
                    # Derive from date + home team
                    # This is approximate - may need refinement
                    date_str = row['gameday'].replace('-', '')
                    # Add approximate hour based on game time
                    ngs_ids.append(f"{date_str}00")  # Placeholder
            
            return ngs_ids
            
        except ImportError:
            logger.warning("nfl_data_py not available, cannot get game IDs")
            return []
    
    def scrape_game(self, game_id: str) -> Optional[Dict]:
        """
        Scrape a single game from Game Center.
        
        Args:
            game_id: NGS game ID (format: YYYYMMDDHH)
            
        Returns:
            Dict with passers, rushers, receivers, etc.
        """
        data = self.client.get_game_center(game_id)
        
        if data:
            # Add game_id to all player records
            for category in ['passers', 'rushers', 'receivers', 'passRushers']:
                if category in data:
                    for player in data[category]:
                        player['gameId'] = game_id
        
        return data
    
    def scrape_season_games(
        self,
        season: int,
        game_ids: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Scrape all games for a season.
        
        Args:
            season: Season year
            game_ids: Optional list of game IDs (if not provided, will try to derive)
            
        Returns:
            Dict with DataFrames for passers, rushers, receivers
        """
        if game_ids is None:
            game_ids = self.get_game_ids_for_season(season)
        
        all_passers = []
        all_rushers = []
        all_receivers = []
        all_pass_rushers = []
        
        for game_id in game_ids:
            logger.info(f"Scraping game: {game_id}")
            
            try:
                data = self.scrape_game(game_id)
                
                if data:
                    if 'passers' in data and data['passers']:
                        all_passers.extend(data['passers'])
                    if 'rushers' in data and data['rushers']:
                        all_rushers.extend(data['rushers'])
                    if 'receivers' in data and data['receivers']:
                        all_receivers.extend(data['receivers'])
                    if 'passRushers' in data and data['passRushers']:
                        all_pass_rushers.extend(data['passRushers'])
                        
            except Exception as e:
                logger.error(f"Error scraping game {game_id}: {e}")
                continue
        
        result = {}
        
        if all_passers:
            result['passers'] = pd.DataFrame(all_passers)
            logger.info(f"Collected {len(all_passers)} passer records")
        
        if all_rushers:
            result['rushers'] = pd.DataFrame(all_rushers)
            logger.info(f"Collected {len(all_rushers)} rusher records")
        
        if all_receivers:
            result['receivers'] = pd.DataFrame(all_receivers)
            logger.info(f"Collected {len(all_receivers)} receiver records")
        
        if all_pass_rushers:
            result['pass_rushers'] = pd.DataFrame(all_pass_rushers)
            logger.info(f"Collected {len(all_pass_rushers)} pass rusher records")
        
        return result

