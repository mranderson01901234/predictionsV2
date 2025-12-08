"""
Generate Phase 3 Features Only (No Training)

This script just generates features without training, so you can monitor progress.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.nfl.generate_all_features import generate_all_features
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Generating Phase 3 features...")
    logger.info("This may take several minutes due to weather API calls")
    
    df = generate_all_features()
    
    logger.info("=" * 60)
    logger.info("FEATURE GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total games: {len(df)}")
    logger.info(f"Total columns: {len(df.columns)}")
    logger.info(f"\nFeature breakdown:")
    
    baseline = len([c for c in df.columns if 'last' in c])
    schedule = len([c for c in df.columns if any(x in c for x in ['rest', 'bye', 'travel', 'divisional', 'primetime'])])
    injury = len([c for c in df.columns if 'injury' in c or 'qb_status' in c or 'oline' in c])
    weather = len([c for c in df.columns if any(x in c for x in ['weather', 'temperature', 'wind', 'precipitation', 'dome'])])
    
    logger.info(f"  Baseline: {baseline}")
    logger.info(f"  Schedule: {schedule}")
    logger.info(f"  Injury: {injury}")
    logger.info(f"  Weather: {weather}")

