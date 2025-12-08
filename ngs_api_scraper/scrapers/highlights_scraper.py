"""
Scraper for NGS highlights endpoint.

Highlights endpoint returns notable plays from games.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd

from .base_client import NGSClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HighlightsScraper:
    """
    Scrape highlight plays from NGS.
    """
    
    def __init__(self, output_dir: str = "data/raw/highlights"):
        self.client = NGSClient()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def scrape_highlights(
        self,
        season: int,
        season_type: str = 'REG',
        week: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Scrape highlights for a season/week.
        
        Args:
            season: Season year
            season_type: 'REG' or 'POST'
            week: Week number (optional)
            limit: Number of highlights to return
            
        Returns:
            List of highlight records
        """
        data = self.client.get_highlights(season, season_type, week, limit)
        
        if not data or 'highlights' not in data:
            return []
        
        records = []
        for highlight in data['highlights']:
            highlight['season'] = season
            highlight['seasonType'] = season_type
            highlight['week'] = week if week else highlight.get('week')
            records.append(highlight)
        
        return records
    
    def scrape_season(
        self,
        season: int,
        season_type: str = 'REG',
        include_weekly: bool = True,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Scrape all highlights for a season.
        
        Returns:
            DataFrame with all highlights
        """
        all_records = []
        
        # Season totals
        season_highlights = self.scrape_highlights(season, season_type, week=None, limit=limit)
        all_records.extend(season_highlights)
        
        # Weekly data
        if include_weekly:
            max_week = 18 if season_type == 'REG' else 4
            
            for week in range(1, max_week + 1):
                week_highlights = self.scrape_highlights(season, season_type, week=week, limit=limit)
                all_records.extend(week_highlights)
        
        if all_records:
            df = pd.DataFrame(all_records)
            output_path = self.output_dir / f"highlights_{season}_{season_type}.parquet"
            df.to_parquet(output_path, index=False)
            logger.info(f"Saved {len(df)} highlights to {output_path}")
            return df
        
        return pd.DataFrame()

