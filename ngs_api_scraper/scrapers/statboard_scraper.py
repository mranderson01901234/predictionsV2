"""
Scraper for NGS statboard endpoints.

These are the primary stats endpoints:
- Passing: CPOE, time to throw, aggressiveness, air yards
- Rushing: RYOE, efficiency, time to LOS
- Receiving: Separation, cushion, YAC above expected

Data is available:
- By season (season totals)
- By week (weekly stats)
- For all qualified players
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd

from .base_client import NGSClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatboardScraper:
    """
    Scrape comprehensive statistics from NGS statboards.
    
    Collects passing, rushing, and receiving stats at both
    season and weekly granularity.
    """
    
    STAT_TYPES = ['passing', 'rushing', 'receiving']
    MIN_SEASON = 2018
    
    def __init__(self, output_dir: str = "data/raw/statboards"):
        self.client = NGSClient()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def scrape_season(
        self,
        season: int,
        season_type: str = 'REG',
        include_weekly: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Scrape all statboard data for a season.
        
        Args:
            season: Season year
            season_type: 'REG' or 'POST'
            include_weekly: Also scrape week-by-week data
            
        Returns:
            Dict mapping stat_type to DataFrame
        """
        all_data = {}
        
        for stat_type in self.STAT_TYPES:
            logger.info(f"Scraping {stat_type} for {season} {season_type}")
            
            records = []
            
            # Season totals
            season_stats = self.client.get_statboard(
                stat_type, season, season_type, week=None
            )
            
            if season_stats and 'stats' in season_stats:
                for player in season_stats['stats']:
                    player['season'] = season
                    player['seasonType'] = season_type
                    player['week'] = 0  # 0 = season total
                    records.append(player)
                
                logger.info(f"  Season totals: {len(season_stats['stats'])} players")
            
            # Weekly data
            if include_weekly:
                max_week = 18 if season_type == 'REG' else 4
                
                for week in range(1, max_week + 1):
                    week_stats = self.client.get_statboard(
                        stat_type, season, season_type, week=week
                    )
                    
                    if week_stats and 'stats' in week_stats:
                        for player in week_stats['stats']:
                            player['season'] = season
                            player['seasonType'] = season_type
                            player['week'] = week
                            records.append(player)
                        
                        logger.info(f"  Week {week}: {len(week_stats['stats'])} players")
                    else:
                        logger.debug(f"  Week {week}: No data")
            
            # Convert to DataFrame
            if records:
                df = pd.DataFrame(records)
                all_data[stat_type] = df
                
                # Save intermediate
                output_path = self.output_dir / stat_type / f"{season}_{season_type}.parquet"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(output_path, index=False)
                logger.info(f"  Saved: {output_path}")
        
        return all_data
    
    def scrape_historical(
        self,
        start_season: int = 2018,
        end_season: Optional[int] = None,
        include_weekly: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Scrape all historical data.
        
        Args:
            start_season: First season to scrape (min 2018)
            end_season: Last season (default: current year)
            include_weekly: Include week-by-week data
            
        Returns:
            Dict mapping stat_type to combined DataFrame
        """
        if end_season is None:
            end_season = datetime.now().year
        
        start_season = max(start_season, self.MIN_SEASON)
        
        all_passing = []
        all_rushing = []
        all_receiving = []
        
        for season in range(start_season, end_season + 1):
            for season_type in ['REG', 'POST']:
                try:
                    data = self.scrape_season(season, season_type, include_weekly)
                    
                    if 'passing' in data:
                        all_passing.append(data['passing'])
                    if 'rushing' in data:
                        all_rushing.append(data['rushing'])
                    if 'receiving' in data:
                        all_receiving.append(data['receiving'])
                        
                except Exception as e:
                    logger.error(f"Error scraping {season} {season_type}: {e}")
                    continue
        
        # Combine all seasons
        combined = {}
        
        if all_passing:
            combined['passing'] = pd.concat(all_passing, ignore_index=True)
            self._save_combined(combined['passing'], 'passing')
        
        if all_rushing:
            combined['rushing'] = pd.concat(all_rushing, ignore_index=True)
            self._save_combined(combined['rushing'], 'rushing')
        
        if all_receiving:
            combined['receiving'] = pd.concat(all_receiving, ignore_index=True)
            self._save_combined(combined['receiving'], 'receiving')
        
        logger.info(f"\n=== Scrape Complete ===")
        logger.info(f"Client stats: {self.client.get_stats()}")
        
        return combined
    
    def _save_combined(self, df: pd.DataFrame, stat_type: str):
        """Save combined DataFrame."""
        output_path = self.output_dir / f"{stat_type}_all.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved combined {stat_type}: {len(df)} records to {output_path}")
    
    def scrape_current_week(self, week: int) -> Dict[str, pd.DataFrame]:
        """
        Scrape just the current week's data (for live updates).
        """
        season = datetime.now().year
        return self.scrape_season(season, 'REG', include_weekly=False)


# Passing stats columns reference
PASSING_COLUMNS = {
    # Player info
    'playerName': 'Player name',
    'position': 'Position (QB)',
    'teamId': 'Team ID',
    
    # Core passing stats
    'attempts': 'Pass attempts',
    'completions': 'Completions',
    'completionPercentage': 'Completion %',
    'passYards': 'Passing yards',
    'passTouchdowns': 'Passing TDs',
    'interceptions': 'Interceptions',
    'passerRating': 'Passer rating',
    
    # NGS advanced metrics
    'avgTimeToThrow': 'Average time to throw (seconds)',
    'avgCompletedAirYards': 'Avg air yards on completions',
    'avgIntendedAirYards': 'Avg air yards on attempts',
    'avgAirYardsDifferential': 'Air yards differential (intended - completed)',
    'avgAirYardsToSticks': 'Avg air yards relative to first down marker',
    'aggressiveness': '% of throws into tight coverage',
    'maxAirDistance': 'Max air distance on completion',
    'avgAirDistance': 'Avg air distance on attempts',
    
    # Expected completion
    'expectedCompletionPercentage': 'Expected completion % based on difficulty',
    'completionPercentageAboveExpectation': 'CPOE - actual minus expected',
    
    # Games
    'gamesPlayed': 'Games played',
}


RUSHING_COLUMNS = {
    'playerName': 'Player name',
    'position': 'Position (RB, QB, WR, etc.)',
    'teamId': 'Team ID',
    
    # Core rushing stats
    'rushAttempts': 'Rush attempts',
    'rushYards': 'Rush yards',
    'rushTouchdowns': 'Rush TDs',
    'avgRushYards': 'Avg yards per carry',
    
    # NGS advanced metrics
    'efficiency': 'Distance traveled / yards gained (lower = more direct)',
    'avgTimeToLos': 'Avg time to reach line of scrimmage',
    'percentAttemptsGteEightDefenders': '% of attempts vs 8+ defenders in box',
    
    # Expected rush yards
    'expectedRushYards': 'Expected yards based on blocking/defense',
    'rushYardsOverExpected': 'Actual - expected (total)',
    'rushYardsOverExpectedPerAtt': 'RYOE per attempt',
    'rushPctOverExpected': 'Rush yards % above expected',
}


RECEIVING_COLUMNS = {
    'playerName': 'Player name',
    'position': 'Position (WR, TE, RB)',
    'teamId': 'Team ID',
    
    # Core receiving stats
    'targets': 'Targets',
    'receptions': 'Receptions',
    'catchPercentage': 'Catch %',
    'yards': 'Receiving yards',
    'recTouchdowns': 'Receiving TDs',
    
    # NGS advanced metrics
    'avgCushion': 'Avg distance from DB at snap',
    'avgSeparation': 'Avg separation at catch/incompletion',
    'avgIntendedAirYards': 'Avg depth of target',
    'percentShareOfIntendedAirYards': 'Share of team air yards',
    
    # YAC metrics
    'avgYAC': 'Avg yards after catch',
    'avgExpectedYAC': 'Expected YAC based on situation',
    'avgYACAboveExpectation': 'YAC above expected',
}


if __name__ == "__main__":
    scraper = StatboardScraper()
    
    # Test with one season
    data = scraper.scrape_season(2024, 'REG', include_weekly=True)
    
    for stat_type, df in data.items():
        print(f"\n{stat_type.upper()}: {len(df)} records")
        print(df.columns.tolist())

