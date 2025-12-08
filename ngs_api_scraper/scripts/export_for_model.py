"""
Export NGS data in format ready for model integration.

Outputs:
- Parquet files with clean schema
- Player ID mapping for joining with other data
- Weekly aggregates for feature engineering
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_passing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize passing data."""
    
    # Rename columns for consistency with nflverse
    rename_map = {
        'playerName': 'player_display_name',
        'teamId': 'team_id',
        'avgTimeToThrow': 'avg_time_to_throw',
        'completionPercentageAboveExpectation': 'cpoe',
        'avgCompletedAirYards': 'avg_completed_air_yards',
        'avgIntendedAirYards': 'avg_intended_air_yards',
        'avgAirYardsDifferential': 'avg_air_yards_differential',
        'avgAirYardsToSticks': 'avg_air_yards_to_sticks',
        'passerRating': 'passer_rating',
        'expectedCompletionPercentage': 'expected_completion_percentage',
    }
    
    df = df.rename(columns=rename_map)
    
    # Convert types
    numeric_cols = [
        'avg_time_to_throw', 'cpoe', 'avg_completed_air_yards',
        'avg_intended_air_yards', 'aggressiveness', 'passer_rating'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def clean_rushing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize rushing data."""
    
    rename_map = {
        'playerName': 'player_display_name',
        'teamId': 'team_id',
        'rushAttempts': 'rush_attempts',
        'rushYards': 'rush_yards',
        'rushTouchdowns': 'rush_touchdowns',
        'avgRushYards': 'avg_rush_yards',
        'avgTimeToLos': 'avg_time_to_los',
        'percentAttemptsGteEightDefenders': 'pct_8_defenders',
        'expectedRushYards': 'expected_rush_yards',
        'rushYardsOverExpected': 'ryoe',
        'rushYardsOverExpectedPerAtt': 'ryoe_per_att',
    }
    
    df = df.rename(columns=rename_map)
    
    return df


def clean_receiving_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize receiving data."""
    
    rename_map = {
        'playerName': 'player_display_name',
        'teamId': 'team_id',
        'avgCushion': 'avg_cushion',
        'avgSeparation': 'avg_separation',
        'avgYAC': 'avg_yac',
        'avgExpectedYAC': 'avg_expected_yac',
        'avgYACAboveExpectation': 'yac_above_expected',
        'catchPercentage': 'catch_percentage',
        'percentShareOfIntendedAirYards': 'air_yards_share',
        'recTouchdowns': 'rec_touchdowns',
    }
    
    df = df.rename(columns=rename_map)
    
    return df


def export_for_model(
    input_dir: str = "data/raw/statboards",
    output_dir: str = "data/processed/ngs"
):
    """
    Export cleaned NGS data for model integration.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each stat type
    for stat_type in ['passing', 'rushing', 'receiving']:
        input_file = input_path / f'{stat_type}_all.parquet'
        
        if not input_file.exists():
            logger.warning(f"File not found: {input_file}")
            continue
        
        df = pd.read_parquet(input_file)
        logger.info(f"Loaded {stat_type}: {len(df)} records")
        
        # Clean data
        if stat_type == 'passing':
            df = clean_passing_data(df)
        elif stat_type == 'rushing':
            df = clean_rushing_data(df)
        elif stat_type == 'receiving':
            df = clean_receiving_data(df)
        
        # Save cleaned data
        output_file = output_path / f'{stat_type}_clean.parquet'
        df.to_parquet(output_file, index=False)
        logger.info(f"Saved: {output_file}")
        
        # Also save summary
        logger.info(f"\n{stat_type.upper()} Summary:")
        logger.info(f"  Seasons: {sorted(df['season'].unique())}")
        logger.info(f"  Records: {len(df)}")
        logger.info(f"  Columns: {list(df.columns)}")


if __name__ == "__main__":
    export_for_model()

