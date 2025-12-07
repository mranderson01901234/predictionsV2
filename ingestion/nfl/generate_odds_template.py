"""
Generate Template CSV for NFL Odds Data

Creates a template CSV file that can be populated with historical odds data.
"""

import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ingestion.nfl.schedule import ingest_nfl_schedules


def generate_odds_template(output_path: Path = None):
    """
    Generate a template CSV file for odds data based on games schedule.
    
    Args:
        output_path: Path to save template CSV. If None, uses default.
    """
    if output_path is None:
        output_path = Path(__file__).parent.parent.parent / "data" / "nfl" / "raw" / "odds_template.csv"
    
    print("Fetching NFL schedules to generate odds template...")
    
    # Fetch games data
    games_df = ingest_nfl_schedules()
    
    # Create template with required columns
    template_df = pd.DataFrame({
        "season": games_df["season"],
        "week": games_df["week"],
        "away_team": games_df["away_team"],
        "home_team": games_df["home_team"],
        "close_spread": None,  # To be filled
        "close_total": None,  # To be filled
        "open_spread": None,  # Optional
        "open_total": None,  # Optional
    })
    
    # Save template
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template_df.to_csv(output_path, index=False)
    
    print(f"Generated odds template with {len(template_df)} games")
    print(f"Template saved to: {output_path}")
    print("\nPlease fill in the following columns:")
    print("  - close_spread: Closing point spread (from home team perspective)")
    print("  - close_total: Closing over/under total")
    print("  - open_spread: Opening point spread (optional)")
    print("  - open_total: Opening over/under total (optional)")
    print("\nSpread convention:")
    print("  - Negative spread = home team favored")
    print("  - Positive spread = away team favored")
    
    return template_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate NFL odds template CSV")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for template CSV",
        default=None,
    )
    
    args = parser.parse_args()
    generate_odds_template(args.output)

