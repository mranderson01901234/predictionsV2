"""
Master script to ingest all NFL advanced data sources.

Combines:
- NGS (Next Gen Stats): Player tracking metrics (2016+)
- FTN (Football Technology Network): Play charting data (2022+)
- PFR (Pro Football Reference): Advanced box score stats (2018+)

Usage:
    # Ingest all data sources
    python -m ingestion.nfl.advanced_stats

    # Ingest specific seasons
    python -m ingestion.nfl.advanced_stats --seasons 2020 2021 2022 2023 2024

    # Force refresh (re-download all)
    python -m ingestion.nfl.advanced_stats --force-refresh

    # NGS only
    python -m ingestion.nfl.advanced_stats --ngs-only
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ingest_all_advanced_stats(
    seasons: Optional[List[int]] = None,
    force_refresh: bool = False,
    ngs_only: bool = False,
    ftn_only: bool = False,
    pfr_only: bool = False,
) -> Dict[str, Any]:
    """
    Ingest all NFL advanced data sources.

    Args:
        seasons: List of seasons to ingest (default: all available)
        force_refresh: Force re-download of all data
        ngs_only: Only ingest NGS data
        ftn_only: Only ingest FTN data
        pfr_only: Only ingest PFR data

    Returns:
        Dict with ingested data organized by source
    """
    from ingestion.nfl.ngs import NGSIngester
    from ingestion.nfl.ftn import FTNIngester
    from ingestion.nfl.pfr import PFRIngester

    if seasons is None:
        current_year = datetime.now().year
        # Start from 2016 (NGS availability) to current
        seasons = list(range(2016, current_year + 1))

    logger.info("=== NFL Advanced Data Ingestion ===")
    logger.info(f"Seasons: {seasons}")
    logger.info(f"Force refresh: {force_refresh}")

    results = {}

    # NGS Data (2016+)
    if not ftn_only and not pfr_only:
        logger.info("\n--- Ingesting NGS Data (2016+) ---")
        try:
            ngs = NGSIngester()
            results['ngs'] = ngs.ingest_all(seasons, force_refresh)
            logger.info(f"NGS ingestion complete: {sum(len(df) for df in results['ngs'].values())} total records")
        except Exception as e:
            logger.error(f"NGS ingestion failed: {e}")
            results['ngs'] = {}

    # FTN Data (2022+)
    if not ngs_only and not pfr_only:
        logger.info("\n--- Ingesting FTN Data (2022+) ---")
        try:
            ftn = FTNIngester()
            results['ftn'] = ftn.ingest(seasons, force_refresh)
            logger.info(f"FTN ingestion complete: {len(results['ftn'])} records")
        except Exception as e:
            logger.error(f"FTN ingestion failed: {e}")
            results['ftn'] = pd.DataFrame()

    # PFR Data (2018+)
    if not ngs_only and not ftn_only:
        logger.info("\n--- Ingesting PFR Data (2018+) ---")
        try:
            pfr = PFRIngester()
            results['pfr_weekly'] = pfr.ingest_weekly(seasons, force_refresh)
            results['pfr_seasonal'] = pfr.ingest_seasonal(seasons, force_refresh)
            logger.info(f"PFR ingestion complete")
        except Exception as e:
            logger.error(f"PFR ingestion failed: {e}")
            results['pfr_weekly'] = {}
            results['pfr_seasonal'] = {}

    return results


def print_summary(results: Dict[str, Any]) -> None:
    """Print summary of ingested data."""
    print("\n" + "=" * 60)
    print("=== Ingestion Summary ===")
    print("=" * 60)

    # NGS Summary
    if 'ngs' in results and results['ngs']:
        print("\n--- NGS (Next Gen Stats) ---")
        for stat_type, df in results['ngs'].items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                seasons = sorted(df['season'].unique()) if 'season' in df.columns else []
                print(f"  {stat_type}: {len(df):,} records, seasons {seasons}")
            else:
                print(f"  {stat_type}: No data")

    # FTN Summary
    if 'ftn' in results:
        print("\n--- FTN (Charting Data) ---")
        df = results['ftn']
        if isinstance(df, pd.DataFrame) and not df.empty:
            seasons = sorted(df['season'].unique()) if 'season' in df.columns else []
            games = df['game_id'].nunique() if 'game_id' in df.columns else 0
            print(f"  Plays: {len(df):,}, Games: {games:,}, Seasons: {seasons}")
        else:
            print("  No data")

    # PFR Summary
    if 'pfr_weekly' in results and results['pfr_weekly']:
        print("\n--- PFR Weekly Stats ---")
        for stat_type, df in results['pfr_weekly'].items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                seasons = sorted(df['season'].unique()) if 'season' in df.columns else []
                print(f"  {stat_type}: {len(df):,} records, seasons {seasons}")

    if 'pfr_seasonal' in results and results['pfr_seasonal']:
        print("\n--- PFR Seasonal Stats ---")
        for stat_type, df in results['pfr_seasonal'].items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                seasons = sorted(df['season'].unique()) if 'season' in df.columns else []
                print(f"  {stat_type}: {len(df):,} records, seasons {seasons}")

    print("\n" + "=" * 60)


def get_data_availability() -> Dict[str, Dict]:
    """
    Get summary of what data is available in cache.

    Returns:
        Dict with availability info for each data source
    """
    from ingestion.nfl.ngs import NGSIngester
    from ingestion.nfl.ftn import FTNIngester
    from ingestion.nfl.pfr import PFRIngester

    availability = {
        'ngs': NGSIngester().get_summary(),
        'ftn': FTNIngester().get_summary(),
        'pfr': PFRIngester().get_summary(),
    }

    return availability


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest NFL advanced data (NGS, FTN, PFR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data Sources:
  NGS (Next Gen Stats): Player tracking metrics from 2016+
    - Passing: CPOE, time_to_throw, aggressiveness
    - Rushing: RYOE (rush yards over expected)
    - Receiving: separation, cushion, YAC above expected

  FTN (Football Technology Network): Play charting from 2022+
    - Play type flags (play action, RPO, screen)
    - Pressure and blitz data
    - QB accuracy assessments

  PFR (Pro Football Reference): Advanced box scores from 2018+
    - Pressure rates
    - On-target throw rate
    - Yards before/after contact

Examples:
  # Ingest all data
  python -m ingestion.nfl.advanced_stats

  # Specific seasons
  python -m ingestion.nfl.advanced_stats --seasons 2022 2023 2024

  # NGS only
  python -m ingestion.nfl.advanced_stats --ngs-only

  # Check cached data
  python -m ingestion.nfl.advanced_stats --status
        """
    )

    parser.add_argument('--seasons', nargs='+', type=int, default=None,
                        help="Seasons to ingest (default: all available)")
    parser.add_argument('--force-refresh', action='store_true',
                        help="Force re-download of all data")
    parser.add_argument('--ngs-only', action='store_true',
                        help="Only ingest NGS data")
    parser.add_argument('--ftn-only', action='store_true',
                        help="Only ingest FTN data")
    parser.add_argument('--pfr-only', action='store_true',
                        help="Only ingest PFR data")
    parser.add_argument('--status', action='store_true',
                        help="Show status of cached data (no download)")

    args = parser.parse_args()

    if args.status:
        # Show cached data status
        print("\n=== Cached Data Status ===\n")
        availability = get_data_availability()

        print("NGS (Next Gen Stats):")
        for stat_type, info in availability['ngs'].items():
            print(f"  {stat_type}: {info['records']:,} records, {info['players']} players")
            if info['seasons']:
                print(f"    Seasons: {info['seasons']}")

        print("\nFTN (Charting):")
        info = availability['ftn']
        print(f"  Plays: {info['records']:,}, Games: {info['games']:,}")
        if info['seasons']:
            print(f"    Seasons: {info['seasons']}")

        print("\nPFR (Advanced Stats):")
        for stat_type, info in availability['pfr'].items():
            print(f"  {stat_type}: {info['records']:,} records")
            if info['seasons']:
                print(f"    Seasons: {info['seasons']}")

        return

    # Run ingestion
    results = ingest_all_advanced_stats(
        seasons=args.seasons,
        force_refresh=args.force_refresh,
        ngs_only=args.ngs_only,
        ftn_only=args.ftn_only,
        pfr_only=args.pfr_only,
    )

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
