"""
Scraper for NGS leaders endpoints (top plays).

These endpoints return the top plays in various categories:
- Fastest ball carriers
- Longest runs
- Longest tackles
- Fastest sacks
- Improbable completions
- YAC above expected
- Remarkable rushes
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd

from .base_client import NGSClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LeadersScraper:
    """
    Scrape top plays from NGS leaders endpoints.
    """
    
    LEADER_TYPES = {
        'fastest_ball_carriers': 'speed/ballCarrier',
        'longest_ball_carrier_runs': 'distance/ballCarrier',
        'longest_tackles': 'distance/tackle',
        'fastest_sacks': 'time/sack',
        'improbable_completions': 'expectation/completion/season',
        'yac_above_expected': 'expectation/yac/season',
        'remarkable_rushes': 'expectation/ery/season',
    }
    
    def __init__(self, output_dir: str = "data/raw/leaders"):
        self.client = NGSClient()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def scrape_category(
        self,
        category: str,
        season: int,
        season_type: str = 'REG',
        week: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Scrape a specific leader category.
        
        Args:
            category: Category name (e.g., 'fastest_ball_carriers')
            season: Season year
            season_type: 'REG' or 'POST'
            week: Week number (optional)
            limit: Number of records to return
            
        Returns:
            List of leader records
        """
        if category not in self.LEADER_TYPES:
            raise ValueError(f"Unknown category: {category}")
        
        leader_type = self.LEADER_TYPES[category]
        data = self.client.get_leaders(leader_type, season, season_type, week)
        
        if not data:
            return []
        
        # Different endpoints use different keys
        leaders_key = None
        if 'leaders' in data:
            leaders_key = 'leaders'
        elif 'completionLeaders' in data:
            leaders_key = 'completionLeaders'
        elif 'yacLeaders' in data:
            leaders_key = 'yacLeaders'
        elif 'eryLeaders' in data:
            leaders_key = 'eryLeaders'
        
        if not leaders_key:
            logger.warning(f"No leaders found in response for {category}")
            return []
        
        records = []
        for leader_record in data[leaders_key][:limit]:
            # Flatten the record
            flat_record = {
                'category': category,
                'season': season,
                'seasonType': season_type,
                'week': week if week else data.get('week'),
            }
            
            # Extract leader info
            leader = leader_record.get('leader', {})
            for key, value in leader.items():
                flat_record[f'leader_{key}'] = value
            
            # Extract play info
            play = leader_record.get('play', {})
            for key, value in play.items():
                flat_record[f'play_{key}'] = value
            
            # Extract top-level metrics
            for key in ['time', 'expectation', 'actual', 'difference']:
                if key in leader_record:
                    flat_record[key] = leader_record[key]
            
            records.append(flat_record)
        
        return records
    
    def scrape_all_categories(
        self,
        season: int,
        season_type: str = 'REG',
        week: Optional[int] = None,
        limit: int = 100
    ) -> Dict[str, List[Dict]]:
        """
        Scrape all leader categories.
        
        Returns:
            Dict mapping category name to list of records
        """
        results = {}
        
        for category in self.LEADER_TYPES.keys():
            logger.info(f"Scraping category: {category}")
            records = self.scrape_category(category, season, season_type, week, limit)
            results[category] = records
            logger.info(f"  Extracted {len(records)} records")
        
        return results
    
    def scrape_season(
        self,
        season: int,
        season_type: str = 'REG',
        include_weekly: bool = True,
        limit: int = 100
    ) -> Dict[str, pd.DataFrame]:
        """
        Scrape all leader categories for a season.
        
        Args:
            season: Season year
            season_type: 'REG' or 'POST'
            include_weekly: Also scrape week-by-week data
            limit: Number of records per category
            
        Returns:
            Dict mapping category to DataFrame
        """
        all_data = {}
        
        # Season totals
        season_data = self.scrape_all_categories(season, season_type, week=None, limit=limit)
        
        for category, records in season_data.items():
            all_data[category] = records
        
        # Weekly data
        if include_weekly:
            max_week = 18 if season_type == 'REG' else 4
            
            for week in range(1, max_week + 1):
                week_data = self.scrape_all_categories(season, season_type, week=week, limit=limit)
                
                for category, records in week_data.items():
                    if category in all_data:
                        all_data[category].extend(records)
                    else:
                        all_data[category] = records
        
        # Convert to DataFrames
        result = {}
        for category, records in all_data.items():
            if records:
                df = pd.DataFrame(records)
                result[category] = df
                
                # Save
                output_path = self.output_dir / f"{category}_{season}_{season_type}.parquet"
                df.to_parquet(output_path, index=False)
                logger.info(f"Saved {category}: {len(df)} records to {output_path}")
        
        return result

