"""
Phase 2 Complete Pipeline

Runs feature generation for Phase 2 (baseline + EPA features).
"""

import sys
from pathlib import Path
import time
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from orchestration.pipelines.feature_pipeline import run_phase2_feature_pipeline
from features.feature_table_registry import get_feature_table_path, validate_feature_table_exists
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def validate_phase2_output(df: pd.DataFrame, output_path: Path) -> None:
    """
    Validate Phase 2 feature table output.
    
    Args:
        df: Generated feature dataframe
        output_path: Path where features were saved
    """
    logger.info("\n" + "=" * 60)
    logger.info("Validating Phase 2 Feature Table")
    logger.info("=" * 60)
    
    # Check required columns
    required_cols = ["game_id", "season", "week", "home_team", "away_team"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    logger.info(f"✓ Required columns present: {required_cols}")
    
    # Check for duplicates
    duplicates = df["game_id"].duplicated().sum()
    if duplicates > 0:
        raise ValueError(f"Found {duplicates} duplicate game_ids")
    logger.info(f"✓ No duplicate game_ids ({len(df)} unique games)")
    
    # Check for null team merges (home/away teams should always match)
    null_home = df["home_team"].isna().sum()
    null_away = df["away_team"].isna().sum()
    if null_home > 0 or null_away > 0:
        logger.warning(f"Found {null_home} null home_team, {null_away} null away_team")
    else:
        logger.info("✓ No null team values")
    
    # Check EPA feature columns exist
    epa_cols = [col for col in df.columns if "epa" in col.lower()]
    logger.info(f"✓ Found {len(epa_cols)} EPA-related feature columns")
    
    # Check file exists
    if not output_path.exists():
        raise FileNotFoundError(f"Output file not found: {output_path}")
    logger.info(f"✓ Output file exists: {output_path}")
    
    logger.info("=" * 60)


def run_phase2():
    """Run complete Phase 2 pipeline."""
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info("Phase 2: Baseline + EPA Feature Generation")
    logger.info("=" * 60)
    
    # Get output path from registry
    output_path = get_feature_table_path("phase2")
    logger.info(f"\nOutput will be written to: {output_path}")
    
    # Run pipeline
    logger.info("\nStarting feature generation...")
    df = run_phase2_feature_pipeline()
    
    # Validate output
    validate_phase2_output(df, output_path)
    
    # Summary
    elapsed_time = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2 Complete!")
    logger.info("=" * 60)
    logger.info(f"\nSummary:")
    logger.info(f"  Rows: {len(df):,} games")
    logger.info(f"  Columns: {len(df.columns)} total features")
    logger.info(f"  Generation time: {elapsed_time:.2f} seconds")
    logger.info(f"  Output: {output_path}")
    
    # Verify registry can find it
    try:
        validate_feature_table_exists("phase2")
        logger.info(f"  ✓ Registered with feature_table_registry")
    except FileNotFoundError:
        logger.warning(f"  ⚠ Registry validation failed (file may not be indexed yet)")
    
    return df


if __name__ == "__main__":
    run_phase2()

