"""
Parquet file storage utilities for NGS data.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd

logger = logging.getLogger(__name__)


class ParquetStore:
    """
    Simple parquet file storage manager.
    """
    
    def __init__(self, base_dir: str = "data/processed"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save(
        self,
        df: pd.DataFrame,
        name: str,
        subdirectory: Optional[str] = None
    ) -> Path:
        """
        Save DataFrame to parquet file.
        
        Args:
            df: DataFrame to save
            name: Filename (without extension)
            subdirectory: Optional subdirectory within base_dir
            
        Returns:
            Path to saved file
        """
        if subdirectory:
            output_dir = self.base_dir / subdirectory
        else:
            output_dir = self.base_dir
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{name}.parquet"
        
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(df)} records to {output_path}")
        
        return output_path
    
    def load(
        self,
        name: str,
        subdirectory: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load DataFrame from parquet file.
        
        Args:
            name: Filename (without extension)
            subdirectory: Optional subdirectory within base_dir
            
        Returns:
            DataFrame or None if file doesn't exist
        """
        if subdirectory:
            file_path = self.base_dir / subdirectory / f"{name}.parquet"
        else:
            file_path = self.base_dir / f"{name}.parquet"
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None
        
        try:
            df = pd.read_parquet(file_path)
            logger.info(f"Loaded {len(df)} records from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def list_files(self, subdirectory: Optional[str] = None) -> List[Path]:
        """
        List all parquet files in directory.
        
        Args:
            subdirectory: Optional subdirectory within base_dir
            
        Returns:
            List of file paths
        """
        if subdirectory:
            directory = self.base_dir / subdirectory
        else:
            directory = self.base_dir
        
        if not directory.exists():
            return []
        
        return list(directory.glob("*.parquet"))



