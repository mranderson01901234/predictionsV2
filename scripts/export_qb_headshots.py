"""
Export QB headshot URLs from nfl_data_py for frontend use.

This script extracts quarterback headshot URLs from NFL roster data
and exports them to JSON format for use in the frontend.
"""

import pandas as pd
from pathlib import Path
import json
import logging
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_qb_headshots(season: int = 2024, output_path: Path = None):
    """
    Extract QB headshot URLs from nfl_data_py and export to JSON.
    
    Args:
        season: NFL season year to fetch rosters for
        output_path: Path to save JSON file. If None, saves to web/public/qb_images.json
    """
    try:
        import nfl_data_py as nfl
    except ImportError:
        raise ImportError(
            "nfl_data_py is required. Install with: pip install nfl-data-py"
        )
    
    logger.info(f"Fetching roster data for season {season}...")
    
    # Get roster data which includes headshot_url
    rosters = nfl.import_seasonal_rosters([season])
    
    logger.info(f"Fetched {len(rosters)} roster entries")
    
    # Filter for quarterbacks
    qbs = rosters[rosters['position'] == 'QB'][['player_name', 'headshot_url', 'team']].copy()
    
    # Remove duplicates (same QB on multiple teams/rosters)
    # Keep the most recent entry (assuming rosters are ordered by date)
    qbs = qbs.drop_duplicates(subset=['player_name'], keep='last')
    
    logger.info(f"Found {len(qbs)} unique quarterbacks")
    
    # Clean up the data
    # Fill NaN headshot_urls with empty string
    qbs['headshot_url'] = qbs['headshot_url'].fillna('')
    
    # Sort by player name for easier browsing
    qbs = qbs.sort_values('player_name').reset_index(drop=True)
    
    # Set output path
    if output_path is None:
        output_path = project_root / 'web' / 'public' / 'qb_images.json'
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export to JSON
    # Use orient='records' to get array of objects
    qbs_dict = qbs.to_dict(orient='records')
    
    with open(output_path, 'w') as f:
        json.dump(qbs_dict, f, indent=2)
    
    logger.info(f"Exported {len(qbs_dict)} QB headshots to {output_path}")
    
    # Print summary
    qbs_with_images = qbs[qbs['headshot_url'] != '']
    logger.info(f"QBs with headshot URLs: {len(qbs_with_images)}/{len(qbs)}")
    
    if len(qbs_with_images) > 0:
        logger.info("Sample QBs with headshots:")
        for _, qb in qbs_with_images.head(5).iterrows():
            logger.info(f"  - {qb['player_name']} ({qb['team']}): {qb['headshot_url'][:50]}...")
    
    return qbs


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export QB headshot URLs to JSON")
    parser.add_argument(
        "--season",
        type=int,
        default=2024,
        help="NFL season year (default: 2024)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: web/public/qb_images.json)"
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output) if args.output else None
    
    try:
        qbs_df = export_qb_headshots(season=args.season, output_path=output_path)
        print(f"\n✓ Successfully exported {len(qbs_df)} QB headshots")
    except Exception as e:
        logger.error(f"Error exporting QB headshots: {e}")
        sys.exit(1)

