"""
Scraper for NGS charts endpoint.

Charts endpoint returns pass/route/carry visualization data.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd

from .base_client import NGSClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChartsScraper:
    """
    Scrape chart data from NGS.
    """
    
    def __init__(self, output_dir: str = "data/raw/charts"):
        self.client = NGSClient()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def scrape_charts(
        self,
        season: int,
        chart_type: str = 'all',
        season_type: str = 'REG',
        week: Optional[int] = None,
        team_id: str = 'all',
        player_id: str = 'all',
        count: int = 100
    ) -> List[Dict]:
        """
        Scrape charts for a season/week.
        
        Args:
            season: Season year
            chart_type: 'pass', 'route', 'carry', or 'all'
            season_type: 'REG' or 'POST'
            week: Week number (optional)
            team_id: Team ID or 'all'
            player_id: Player ESB ID or 'all'
            count: Number of charts to return
            
        Returns:
            List of chart records
        """
        data = self.client.get_charts(
            season, chart_type, season_type, week, team_id, player_id, count
        )
        
        if not data or 'charts' not in data:
            return []
        
        records = []
        for chart in data['charts']:
            chart['season'] = season
            chart['seasonType'] = season_type
            chart['week'] = week if week else chart.get('week')
            chart['chartType'] = chart_type
            records.append(chart)
        
        return records
    
    def scrape_season(
        self,
        season: int,
        season_type: str = 'REG',
        include_weekly: bool = True,
        count: int = 100
    ) -> pd.DataFrame:
        """
        Scrape all charts for a season.
        
        Returns:
            DataFrame with all charts
        """
        all_records = []
        
        chart_types = ['pass', 'route', 'carry', 'all']
        
        for chart_type in chart_types:
            logger.info(f"Scraping {chart_type} charts for {season} {season_type}")
            
            # Season totals
            season_charts = self.scrape_charts(
                season, chart_type, season_type, week=None, count=count
            )
            all_records.extend(season_charts)
            
            # Weekly data
            if include_weekly:
                max_week = 18 if season_type == 'REG' else 4
                
                for week in range(1, max_week + 1):
                    week_charts = self.scrape_charts(
                        season, chart_type, season_type, week=week, count=count
                    )
                    all_records.extend(week_charts)
        
        if all_records:
            df = pd.DataFrame(all_records)
            output_path = self.output_dir / f"charts_{season}_{season_type}.parquet"
            df.to_parquet(output_path, index=False)
            logger.info(f"Saved {len(df)} charts to {output_path}")
            return df
        
        return pd.DataFrame()

