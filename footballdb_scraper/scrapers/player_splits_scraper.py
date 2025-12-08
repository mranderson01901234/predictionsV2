"""
Scraper for FootballDB player splits.

Scrapes situational statistics for players:
- Home/Away splits
- Surface splits (Grass/Turf)
- Score differential splits
- Quarter/Half splits
- Day of week splits
"""
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
import pandas as pd
import yaml

from scrapers.base_scraper import FootballDBScraper
from parsers.stats_table_parser import StatsTableParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlayerSplitsScraper:
    """
    Scrape player situational statistics from FootballDB.
    """
    
    BASE_URL = "https://www.footballdb.com/statistics/nfl/player-splits"
    
    # All stat types
    STAT_TYPES = ['passing', 'rushing', 'receiving', 'kicking']
    
    # Priority splits for model (HIGH VALUE)
    PRIORITY_SPLITS = [
        # Surface (very predictive)
        'surface-grass',
        'surface-turf',
        
        # Score differential (clutch performance)
        'trailing-by-1-to-8',
        'leading-by-1-to-8',
        'tied-games',
        
        # Quarter (late game performance)
        'fourth-quarter',
        'second-half',
        
        # Home/Away
        'home-games',
        'away-games',
        
        # Day of week (Thursday games different)
        'thursday-games',
        'monday-games',
        'sunday-games',
        
        # Division games (familiarity)
        'division-games',
    ]
    
    # All available splits
    ALL_SPLITS = [
        # Season
        'year-to-date',
        
        # Location
        'home-games', 'away-games',
        
        # Surface
        'surface-grass', 'surface-turf',
        
        # Quarter
        'first-quarter', 'second-quarter', 
        'third-quarter', 'fourth-quarter',
        
        # Half
        'first-half', 'second-half', 'last-two-minutes',
        
        # Score differential
        'leading-by-17-plus', 'leading-by-9-to-16', 'leading-by-1-to-8',
        'tied-games',
        'trailing-by-1-to-8', 'trailing-by-9-to-16', 'trailing-by-17-plus',
        
        # Day of week
        'sunday-games', 'monday-games', 'thursday-games', 'other-days',
        
        # Month
        'september', 'october', 'november', 'december', 'january',
        
        # Division
        'division-games', 'non-division-games',
        
        # Result
        'wins', 'losses',
        
        # Down
        'first-down', 'second-down', 'third-down', 'fourth-down',
    ]
    
    def __init__(self, output_dir: str = "data/raw/footballdb/player_splits"):
        self.scraper = FootballDBScraper()
        self.parser = StatsTableParser()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_split_url(
        self, 
        stat_type: str, 
        split_type: str, 
        year: int
    ) -> str:
        """Construct URL for a specific split page."""
        return f"{self.BASE_URL}/{stat_type}/{split_type}/{year}"
    
    def scrape_split(
        self,
        stat_type: str,
        split_type: str,
        year: int
    ) -> List[Dict[str, Any]]:
        """
        Scrape a single split page.
        
        Args:
            stat_type: 'passing', 'rushing', 'receiving', 'kicking'
            split_type: Split slug (e.g., 'surface-grass')
            year: Season year
            
        Returns:
            List of player records
        """
        url = self.get_split_url(stat_type, split_type, year)
        soup = self.scraper.fetch_and_parse(url)
        
        if not soup:
            logger.warning(f"Failed to fetch: {url}")
            return []
        
        result = self.parser.parse_player_splits_page(soup)
        records = result.get('records', [])
        
        # Add metadata to each record
        for record in records:
            record['stat_type'] = stat_type
            record['split_type'] = split_type
            record['season'] = year
        
        logger.info(f"Scraped {len(records)} records: {stat_type}/{split_type}/{year}")
        return records
    
    def scrape_season(
        self,
        year: int,
        stat_types: Optional[List[str]] = None,
        split_types: Optional[List[str]] = None,
        priority_only: bool = False
    ) -> pd.DataFrame:
        """
        Scrape all splits for a season.
        
        Args:
            year: Season year
            stat_types: List of stat types (default: all)
            split_types: List of split types (default: all)
            priority_only: Only scrape priority splits
            
        Returns:
            DataFrame with all player split records
        """
        if stat_types is None:
            stat_types = self.STAT_TYPES
        
        if split_types is None:
            split_types = self.PRIORITY_SPLITS if priority_only else self.ALL_SPLITS
        
        all_records = []
        
        for stat_type in stat_types:
            for split_type in split_types:
                try:
                    records = self.scrape_split(stat_type, split_type, year)
                    all_records.extend(records)
                except Exception as e:
                    logger.error(f"Error scraping {stat_type}/{split_type}/{year}: {e}")
                    continue
        
        if not all_records:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_records)
        
        # Clean up DataFrame - ensure string columns are properly typed
        # Convert all object columns to string to avoid parquet conversion issues
        for col in df.columns:
            if df[col].dtype == 'object':
                # Replace NaN with empty string, then convert to string
                df[col] = df[col].fillna('').astype(str)
                # Replace empty strings back to None for cleaner data
                df[col] = df[col].replace('', None)
        
        # Save intermediate
        output_path = self.output_dir / f"player_splits_{year}.parquet"
        df.to_parquet(output_path, index=False, engine='pyarrow')
        logger.info(f"Saved {len(df)} records to {output_path}")
        
        return df
    
    def scrape_historical(
        self,
        start_year: int = 2015,
        end_year: int = 2024,
        priority_only: bool = True
    ) -> pd.DataFrame:
        """
        Scrape historical data for multiple seasons.
        
        Args:
            start_year: First season
            end_year: Last season
            priority_only: Only scrape priority splits (recommended)
            
        Returns:
            Combined DataFrame
        """
        all_dfs = []
        
        for year in range(start_year, end_year + 1):
            logger.info(f"\n=== Scraping {year} ===")
            
            try:
                df = self.scrape_season(year, priority_only=priority_only)
                if not df.empty:
                    all_dfs.append(df)
            except Exception as e:
                logger.error(f"Error scraping {year}: {e}")
                continue
        
        if not all_dfs:
            return pd.DataFrame()
        
        combined = pd.concat(all_dfs, ignore_index=True)
        
        # Save combined
        output_path = self.output_dir / "player_splits_all.parquet"
        combined.to_parquet(output_path, index=False)
        
        logger.info(f"\n=== Complete ===")
        logger.info(f"Total records: {len(combined)}")
        logger.info(f"Scraper stats: {self.scraper.get_stats()}")
        
        return combined
    
    def scrape_surface_splits_only(
        self,
        start_year: int = 2015,
        end_year: int = 2024
    ) -> pd.DataFrame:
        """
        Scrape only surface splits (Grass vs Turf).
        
        This is the highest-value split for the model.
        """
        return self.scrape_historical(
            start_year=start_year,
            end_year=end_year,
            split_types=['surface-grass', 'surface-turf']
        )


if __name__ == "__main__":
    scraper = PlayerSplitsScraper()
    
    # Test with current year, priority splits only
    df = scraper.scrape_season(2024, priority_only=True)
    
    print(f"\nScraped {len(df)} records")
    print(f"\nStat types: {df['stat_type'].unique()}")
    print(f"Split types: {df['split_type'].unique()}")

