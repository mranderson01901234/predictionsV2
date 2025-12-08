"""
Parquet storage utilities for FootballDB data.

Provides convenient functions for saving and loading scraped data.
"""
import logging
from pathlib import Path
from typing import Optional, List
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParquetStore:
    """
    Simple parquet storage wrapper.
    
    Provides methods for saving and loading data with consistent paths.
    """
    
    def __init__(self, base_dir: str = "data/raw/footballdb"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save(
        self,
        df: pd.DataFrame,
        filename: str,
        subdirectory: Optional[str] = None
    ):
        """
        Save DataFrame to parquet file.
        
        Args:
            df: DataFrame to save
            filename: Filename (with or without .parquet extension)
            subdirectory: Optional subdirectory within base_dir
        """
        if not filename.endswith('.parquet'):
            filename += '.parquet'
        
        if subdirectory:
            output_dir = self.base_dir / subdirectory
        else:
            output_dir = self.base_dir
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
        
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(df)} records to {output_path}")
    
    def load(
        self,
        filename: str,
        subdirectory: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load DataFrame from parquet file.
        
        Args:
            filename: Filename (with or without .parquet extension)
            subdirectory: Optional subdirectory within base_dir
            
        Returns:
            DataFrame, or empty DataFrame if file doesn't exist
        """
        if not filename.endswith('.parquet'):
            filename += '.parquet'
        
        if subdirectory:
            input_dir = self.base_dir / subdirectory
        else:
            input_dir = self.base_dir
        
        input_path = input_dir / filename
        
        if not input_path.exists():
            logger.warning(f"File not found: {input_path}")
            return pd.DataFrame()
        
        return pd.read_parquet(input_path)
    
    def list_files(self, subdirectory: Optional[str] = None) -> List[str]:
        """
        List all parquet files in directory.
        
        Args:
            subdirectory: Optional subdirectory within base_dir
            
        Returns:
            List of filenames
        """
        if subdirectory:
            target_dir = self.base_dir / subdirectory
        else:
            target_dir = self.base_dir
        
        if not target_dir.exists():
            return []
        
        return [f.name for f in target_dir.glob('*.parquet')]

