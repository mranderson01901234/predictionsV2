"""
FTN (Football Technology Network) Charting Data Ingestion Module

Fetches FTN charting data via nflverse.
FTN manually charts every play with details not captured in official stats.

Available from 2022 onwards, updated within 48 hours of each game.

Key metrics:
- Play type flags: is_play_action, is_screen, is_rpo, is_trick_play
- QB behavior: is_qb_out_of_pocket, is_throw_away, is_batted_pass
- Target details: is_catchable, is_contested, is_drop
- Pressure: is_blitz, n_pass_rushers, n_blitzers, is_qb_hit

Usage:
    from ingestion.nfl.ftn import FTNIngester
    ingester = FTNIngester()
    data = ingester.ingest(seasons=[2022, 2023, 2024])
"""

import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import nflreadpy first (newer), fall back to nfl_data_py
try:
    import nflreadpy as nfl
    USING_NFLREADPY = True
    logger.info("Using nflreadpy for FTN data")
except ImportError:
    try:
        import nfl_data_py as nfl
        USING_NFLREADPY = False
        logger.info("Using nfl_data_py for FTN data")
    except ImportError:
        raise ImportError(
            "Either nflreadpy or nfl_data_py is required. "
            "Install with: pip install nflreadpy or pip install nfl-data-py"
        )


class FTNIngester:
    """
    Download and cache FTN charting data.

    FTN provides manually charted play-by-play details that are not
    available in standard NFL data, including play design, pressure,
    and pass accuracy assessments.
    """

    # FTN data starts in 2022
    MIN_SEASON = 2022

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the FTN ingester.

        Args:
            cache_dir: Directory to cache data. Defaults to data/nfl/raw/ftn
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "data" / "nfl" / "raw" / "ftn"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def ingest(
        self,
        seasons: Optional[List[int]] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Ingest FTN charting data.

        Args:
            seasons: List of seasons (2022+)
            force_refresh: Re-download even if cached

        Returns:
            DataFrame with FTN charting data
        """
        if seasons is None:
            import datetime
            current_year = datetime.datetime.now().year
            seasons = list(range(self.MIN_SEASON, current_year + 1))

        # Validate - FTN only available 2022+
        seasons = [s for s in seasons if s >= self.MIN_SEASON]

        if not seasons:
            logger.warning(f"No valid seasons for FTN data (requires {self.MIN_SEASON}+)")
            return pd.DataFrame()

        cache_path = self.cache_dir / "ftn_data.parquet"

        if cache_path.exists() and not force_refresh:
            logger.info(f"Loading FTN from cache: {cache_path}")
            df = pd.read_parquet(cache_path)

            # Check for missing seasons
            cached_seasons = df['season'].unique().tolist() if 'season' in df.columns else []
            missing_seasons = [s for s in seasons if s not in cached_seasons]

            if missing_seasons:
                logger.info(f"Fetching missing FTN seasons: {missing_seasons}")
                new_df = self._fetch_ftn_data(missing_seasons)
                if not new_df.empty:
                    df = pd.concat([df, new_df], ignore_index=True)
                    # Deduplicate on game_id + play_id
                    dedup_cols = ['game_id', 'play_id'] if all(c in df.columns for c in ['game_id', 'play_id']) else None
                    if dedup_cols:
                        df = df.drop_duplicates(subset=dedup_cols, keep='last')
                    self._save_cache(df, cache_path)
        else:
            df = self._fetch_ftn_data(seasons)
            if not df.empty:
                self._save_cache(df, cache_path)

        logger.info(f"FTN data: {len(df):,} plays")
        return df

    def _fetch_ftn_data(self, seasons: List[int]) -> pd.DataFrame:
        """
        Fetch FTN data from nflverse.

        Args:
            seasons: List of seasons to fetch

        Returns:
            DataFrame with FTN charting data
        """
        try:
            if USING_NFLREADPY:
                df = nfl.load_ftn_charting(seasons=seasons)
                # nflreadpy returns Polars, convert to pandas
                if hasattr(df, 'to_pandas'):
                    df = df.to_pandas()
            else:
                df = nfl.import_ftn_data(seasons)

            logger.info(f"Fetched {len(df)} FTN records")
            return df

        except Exception as e:
            logger.error(f"Error fetching FTN data: {e}")
            return pd.DataFrame()

    def _save_cache(self, df: pd.DataFrame, path: Path) -> None:
        """Save DataFrame to parquet cache."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info(f"Cached FTN to {path}")

    def get_summary(self) -> dict:
        """
        Get summary of cached FTN data.

        Returns:
            Dict with stats about the data
        """
        cache_path = self.cache_dir / "ftn_data.parquet"

        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            return {
                'records': len(df),
                'seasons': sorted(df['season'].unique().tolist()) if 'season' in df.columns else [],
                'columns': list(df.columns),
                'games': df['game_id'].nunique() if 'game_id' in df.columns else 0,
            }
        return {'records': 0, 'seasons': [], 'columns': [], 'games': 0}


# FTN Columns Reference
FTN_COLUMNS = {
    # Identifiers
    'nflverse_game_id': 'Game identifier in nflverse format',
    'game_id': 'Game identifier',
    'play_id': 'Play identifier',
    'season': 'Season year',
    'week': 'Week number',

    # Play type flags
    'is_no_huddle': 'No huddle offense (True/False)',
    'is_motion': 'Pre-snap motion used (True/False)',
    'is_play_action': 'Play action fake (True/False)',
    'is_screen': 'Screen pass (True/False)',
    'is_rpo': 'Run-pass option (True/False)',
    'is_trick_play': 'Trick play (flea flicker, etc.) (True/False)',

    # QB behavior
    'is_qb_out_of_pocket': 'QB left pocket (True/False)',
    'is_throw_away': 'Intentional throwaway (True/False)',
    'is_batted_pass': 'Pass batted at line of scrimmage (True/False)',
    'is_qb_spike': 'Spike to stop clock (True/False)',
    'is_qb_kneel': 'QB kneel (True/False)',

    # Target/pass details
    'is_catchable': 'Pass was catchable (True/False)',
    'is_contested': 'Catch was contested (True/False)',
    'is_drop': 'Receiver dropped catchable pass (True/False)',
    'is_interception_worthy': 'Should have been intercepted (True/False)',
    'is_qb_sneak': 'QB sneak run (True/False)',

    # Pressure and rush
    'is_blitz': 'Defense blitzed (True/False)',
    'n_pass_rushers': 'Number of pass rushers (int)',
    'n_blitzers': 'Number of blitzers (int)',
    'is_qb_hit': 'QB was hit on throw (True/False)',
    'is_sack': 'QB was sacked (True/False)',
    'is_qb_hurry': 'QB was hurried (True/False)',

    # Route and coverage
    'route': 'Route type run by receiver',
    'target_location': 'Where target was on field',
    'n_defenders_in_box': 'Number of defenders in the box',
}

# Feature engineering hints for FTN data
FTN_FEATURE_IDEAS = {
    'play_action_rate': 'Pct of dropbacks with play action - indicates offensive scheme',
    'blitz_rate_faced': 'Pct of dropbacks facing blitz - indicates opponent perception',
    'pressure_rate': 'Pct of dropbacks with pressure - O-line quality',
    'catchable_throw_rate': 'Pct of passes that were catchable - QB accuracy',
    'drop_rate': 'Pct of catchable passes dropped - WR reliability',
    'interception_worthy_rate': 'Pct of passes that should have been INTs - turnover luck',
    'contested_catch_rate': 'Pct of contested catches made - WR ability',
    'rpo_rate': 'Pct of plays that are RPOs - scheme indicator',
    'screen_rate': 'Pct of passes that are screens - scheme indicator',
    'out_of_pocket_rate': 'Pct of dropbacks where QB left pocket - mobility/pressure',
}


def main():
    """Main ingestion script for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest FTN charting data")
    parser.add_argument('--seasons', nargs='+', type=int, default=None,
                        help="Seasons to ingest (default: 2022 to current)")
    parser.add_argument('--force-refresh', action='store_true',
                        help="Force re-download of all data")

    args = parser.parse_args()

    ingester = FTNIngester()
    df = ingester.ingest(args.seasons, args.force_refresh)

    # Print summary
    print("\n=== FTN Data Summary ===")
    if not df.empty:
        print(f"Records: {len(df):,}")
        print(f"Seasons: {sorted(df['season'].unique()) if 'season' in df.columns else 'N/A'}")
        print(f"Games: {df['game_id'].nunique() if 'game_id' in df.columns else 'N/A'}")
        print(f"\nColumns: {list(df.columns)}")

        # Show some boolean column distributions
        bool_cols = [c for c in df.columns if c.startswith('is_')]
        if bool_cols:
            print(f"\n=== Boolean Column Distributions ===")
            for col in bool_cols[:10]:
                if col in df.columns:
                    true_pct = df[col].mean() * 100 if df[col].dtype == bool else df[col].sum() / len(df) * 100
                    print(f"  {col}: {true_pct:.1f}% True")
    else:
        print("No FTN data available")


if __name__ == "__main__":
    main()
