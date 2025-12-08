"""
Data storage for scraped NFL.com data.

Uses Parquet files for efficient storage and fast loading.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class NFLDataStore:
    """
    Store and retrieve scraped NFL.com data.
    """
    
    def __init__(self, data_dir: str = "data/final"):
        # Resolve path relative to nfl_scraper root if not absolute
        if not Path(data_dir).is_absolute():
            # Assume we're running from nfl_scraper directory
            self.data_dir = Path(__file__).parent.parent / data_dir
        else:
            self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    # === INJURIES ===
    
    def save_injuries(self, records: List, filename: str = "injuries.parquet"):
        """Save injury records to Parquet file."""
        if not records:
            logger.warning("No injury records to save")
            return
        
        # Convert to DataFrame
        data = [r.to_dict() if hasattr(r, 'to_dict') else r for r in records]
        df = pd.DataFrame(data)
        
        # Save to Parquet
        path = self.data_dir / filename
        df.to_parquet(path, index=False)
        logger.info(f"Saved {len(df)} injury records to {path}")
    
    def load_injuries(self, filename: str = "injuries.parquet") -> pd.DataFrame:
        """Load injury records from Parquet file."""
        path = self.data_dir / filename
        
        if not path.exists():
            logger.warning(f"Injury file not found: {path}")
            return pd.DataFrame()
        
        return pd.read_parquet(path)
    
    def append_injuries(self, records: List, filename: str = "injuries.parquet"):
        """Append new injury records to existing file."""
        existing = self.load_injuries(filename)
        
        new_data = [r.to_dict() if hasattr(r, 'to_dict') else r for r in records]
        new_df = pd.DataFrame(new_data)
        
        if not existing.empty:
            combined = pd.concat([existing, new_df], ignore_index=True)
            # Deduplicate
            combined = combined.drop_duplicates(
                subset=['season', 'week', 'player_id', 'team'],
                keep='last'
            )
        else:
            combined = new_df
        
        self.save_injuries(combined.to_dict('records'), filename)
    
    # === PLAYER STATS ===
    
    def save_player_stats(
        self,
        stats: Dict,
        stat_type: str,
        filename: Optional[str] = None
    ):
        """
        Save player stats to Parquet.
        
        Args:
            stats: Dict mapping player_slug to stats
            stat_type: "career", "situational", or "game_logs"
            filename: Override default filename
        """
        if filename is None:
            filename = f"player_{stat_type}.parquet"
        
        # Flatten nested stats for storage
        records = []
        
        for player_slug, player_stats in stats.items():
            if stat_type == 'career':
                record = {'player_slug': player_slug, **player_stats.get('career_totals', {})}
                records.append(record)
            elif stat_type == 'situational':
                for season, situational in player_stats.get('situational', {}).items():
                    if situational:
                        record = {
                            'player_slug': player_slug,
                            'season': season,
                            **self._flatten_dict(situational)
                        }
                        records.append(record)
            elif stat_type == 'game_logs':
                for season, logs in player_stats.get('game_logs', {}).items():
                    if logs:
                        for log in logs:
                            records.append(log)
        
        if records:
            df = pd.DataFrame(records)
            path = self.data_dir / filename
            df.to_parquet(path, index=False)
            logger.info(f"Saved {len(df)} {stat_type} records to {path}")
    
    def load_player_stats(
        self,
        stat_type: str,
        filename: Optional[str] = None
    ) -> pd.DataFrame:
        """Load player stats from Parquet."""
        if filename is None:
            filename = f"player_{stat_type}.parquet"
        
        path = self.data_dir / filename
        
        if not path.exists():
            logger.warning(f"Player stats file not found: {path}")
            return pd.DataFrame()
        
        return pd.read_parquet(path)
    
    # === TRANSACTIONS ===
    
    def save_transactions(self, records: List, filename: str = "transactions.parquet"):
        """Save transaction records to Parquet."""
        if not records:
            return
        
        data = [
            {
                'date': r.date.isoformat() if r.date else None,
                'transaction_type': r.transaction_type,
                'from_team': r.from_team,
                'to_team': r.to_team,
                'player_name': r.player_name,
                'player_id': r.player_id,
                'position': r.position,
                'details': r.details,
            }
            for r in records
        ]
        
        df = pd.DataFrame(data)
        path = self.data_dir / filename
        df.to_parquet(path, index=False)
        logger.info(f"Saved {len(df)} transaction records to {path}")
    
    def load_transactions(self, filename: str = "transactions.parquet") -> pd.DataFrame:
        """Load transaction records from Parquet."""
        path = self.data_dir / filename
        
        if not path.exists():
            return pd.DataFrame()
        
        return pd.read_parquet(path)
    
    # === HELPERS ===
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

