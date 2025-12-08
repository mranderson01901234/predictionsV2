"""
NFL Next Gen Stats Data Ingestion Module

Fetches NFL Next Gen Stats (NGS) data via nfl-data-py / nflreadpy.
NGS provides player tracking metrics from RFID chips in shoulder pads.
Data available from 2016 onwards, updated weekly during season.

Key metrics:
- Passing: time_to_throw, CPOE, aggressiveness, air_yards
- Rushing: RYOE (rush yards over expected), efficiency, time_to_LOS
- Receiving: separation, cushion, YAC above expected

Usage:
    from ingestion.nfl.ngs import NGSIngester
    ingester = NGSIngester()
    data = ingester.ingest_all(seasons=[2020, 2021, 2022, 2023, 2024])
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import nflreadpy first (newer), fall back to nfl_data_py
try:
    import nflreadpy as nfl
    USING_NFLREADPY = True
    logger.info("Using nflreadpy for NGS data")
except ImportError:
    try:
        import nfl_data_py as nfl
        USING_NFLREADPY = False
        logger.info("Using nfl_data_py for NGS data")
    except ImportError:
        raise ImportError(
            "Either nflreadpy or nfl_data_py is required. "
            "Install with: pip install nflreadpy or pip install nfl-data-py"
        )


class NGSIngester:
    """
    Download and cache NFL Next Gen Stats data.

    NGS tracks every player on every play at 10 Hz (10 times per second).
    The raw tracking data is proprietary, but aggregated metrics are public.
    """

    # NGS is available from 2016 onwards
    MIN_SEASON = 2016

    # Stat types available
    STAT_TYPES = ['passing', 'rushing', 'receiving']

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the NGS ingester.

        Args:
            cache_dir: Directory to cache data. Defaults to data/nfl/raw/ngs
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "data" / "nfl" / "raw" / "ngs"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def ingest_all(
        self,
        seasons: Optional[List[int]] = None,
        force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Ingest all NGS data for specified seasons.

        Args:
            seasons: List of seasons (default: 2016 to current)
            force_refresh: Re-download even if cached

        Returns:
            Dict mapping stat_type to DataFrame
        """
        if seasons is None:
            import datetime
            current_year = datetime.datetime.now().year
            seasons = list(range(self.MIN_SEASON, current_year + 1))

        # Validate seasons
        seasons = [s for s in seasons if s >= self.MIN_SEASON]

        if not seasons:
            logger.warning(f"No valid seasons provided (NGS requires {self.MIN_SEASON}+)")
            return {}

        all_data = {}

        for stat_type in self.STAT_TYPES:
            logger.info(f"Ingesting NGS {stat_type} data for seasons {seasons}")

            cache_path = self.cache_dir / stat_type / "ngs_data.parquet"

            if cache_path.exists() and not force_refresh:
                logger.info(f"  Loading from cache: {cache_path}")
                df = pd.read_parquet(cache_path)

                # Check if we need to update with new seasons
                cached_seasons = df['season'].unique().tolist() if 'season' in df.columns else []
                missing_seasons = [s for s in seasons if s not in cached_seasons]

                if missing_seasons:
                    logger.info(f"  Fetching missing seasons: {missing_seasons}")
                    new_df = self._fetch_ngs_data(stat_type, missing_seasons)
                    if not new_df.empty:
                        df = pd.concat([df, new_df], ignore_index=True)
                        df = df.drop_duplicates(
                            subset=['season', 'week', 'player_gsis_id'] if 'player_gsis_id' in df.columns else ['season', 'week'],
                            keep='last'
                        )
                        self._save_cache(df, cache_path)
            else:
                df = self._fetch_ngs_data(stat_type, seasons)
                if not df.empty:
                    self._save_cache(df, cache_path)

            all_data[stat_type] = df
            if not df.empty:
                logger.info(f"  {stat_type}: {len(df)} records, seasons {df['season'].min()}-{df['season'].max()}")
            else:
                logger.warning(f"  {stat_type}: No data fetched")

        return all_data

    def _fetch_ngs_data(self, stat_type: str, seasons: List[int]) -> pd.DataFrame:
        """
        Fetch NGS data from nflverse.

        Args:
            stat_type: One of 'passing', 'rushing', 'receiving'
            seasons: List of seasons to fetch

        Returns:
            DataFrame with NGS data
        """
        try:
            if USING_NFLREADPY:
                # nflreadpy API
                df = nfl.load_nextgen_stats(seasons=seasons, stat_type=stat_type)
                # nflreadpy returns Polars, convert to pandas
                if hasattr(df, 'to_pandas'):
                    df = df.to_pandas()
            else:
                # nfl_data_py API
                df = nfl.import_ngs_data(stat_type, seasons)

            logger.info(f"  Fetched {len(df)} {stat_type} records")
            return df

        except Exception as e:
            logger.error(f"Error fetching NGS {stat_type}: {e}")
            return pd.DataFrame()

    def _save_cache(self, df: pd.DataFrame, path: Path) -> None:
        """Save DataFrame to parquet cache."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info(f"  Cached to {path}")

    def get_passing(self, seasons: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Get passing NGS data.

        Args:
            seasons: List of seasons (default: all available)

        Returns:
            DataFrame with passing NGS metrics
        """
        return self.ingest_all(seasons).get('passing', pd.DataFrame())

    def get_rushing(self, seasons: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Get rushing NGS data.

        Args:
            seasons: List of seasons (default: all available)

        Returns:
            DataFrame with rushing NGS metrics
        """
        return self.ingest_all(seasons).get('rushing', pd.DataFrame())

    def get_receiving(self, seasons: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Get receiving NGS data.

        Args:
            seasons: List of seasons (default: all available)

        Returns:
            DataFrame with receiving NGS metrics
        """
        return self.ingest_all(seasons).get('receiving', pd.DataFrame())

    def get_summary(self) -> Dict[str, Dict]:
        """
        Get summary of cached NGS data.

        Returns:
            Dict with stats about each data type
        """
        summary = {}

        for stat_type in self.STAT_TYPES:
            cache_path = self.cache_dir / stat_type / "ngs_data.parquet"

            if cache_path.exists():
                df = pd.read_parquet(cache_path)
                summary[stat_type] = {
                    'records': len(df),
                    'seasons': sorted(df['season'].unique().tolist()) if 'season' in df.columns else [],
                    'columns': list(df.columns),
                    'players': df['player_gsis_id'].nunique() if 'player_gsis_id' in df.columns else 0,
                }
            else:
                summary[stat_type] = {'records': 0, 'seasons': [], 'columns': [], 'players': 0}

        return summary


# Key NGS columns reference
NGS_PASSING_COLUMNS = {
    'player_gsis_id': 'Unique player identifier',
    'player_display_name': 'Player name',
    'team_abbr': 'Team abbreviation',
    'season': 'Season year',
    'week': 'Week number (0 for season totals)',
    'avg_time_to_throw': 'Average time from snap to throw (seconds)',
    'avg_completed_air_yards': 'Average air yards on completions',
    'avg_intended_air_yards': 'Average intended air yards on all attempts',
    'avg_air_yards_differential': 'Avg intended - avg completed air yards',
    'aggressiveness': '% of throws into tight coverage (<1 yard separation)',
    'completion_percentage': 'Raw completion percentage',
    'expected_completion_percentage': 'Expected completion % based on difficulty',
    'completion_percentage_above_expectation': 'CPOE - most predictive metric',
    'passer_rating': 'Traditional passer rating',
    'avg_air_yards_to_sticks': 'Avg air yards vs first down marker',
    'max_completed_air_distance': 'Longest completed pass air distance',
    'max_air_distance': 'Longest attempted pass air distance',
    'attempts': 'Pass attempts',
    'pass_yards': 'Total passing yards',
    'pass_touchdowns': 'Passing touchdowns',
    'interceptions': 'Interceptions thrown',
}

NGS_RUSHING_COLUMNS = {
    'player_gsis_id': 'Unique player identifier',
    'player_display_name': 'Player name',
    'team_abbr': 'Team abbreviation',
    'season': 'Season year',
    'week': 'Week number (0 for season totals)',
    'efficiency': 'Yards gained / total yards traveled (lower = more north-south)',
    'percent_attempts_gte_eight_defenders': '% of rushes vs stacked boxes',
    'avg_time_to_los': 'Average time to reach line of scrimmage',
    'rush_yards': 'Total rushing yards',
    'rush_attempts': 'Total rush attempts',
    'avg_rush_yards': 'Average yards per rush',
    'expected_rush_yards': 'Expected yards based on blocking/defense',
    'rush_yards_over_expected': 'Total RYOE',
    'rush_yards_over_expected_per_att': 'RYOE per attempt - key skill metric',
    'rush_pct_over_expected': 'Percent over expected',
    'rush_touchdowns': 'Rushing touchdowns',
}

NGS_RECEIVING_COLUMNS = {
    'player_gsis_id': 'Unique player identifier',
    'player_display_name': 'Player name',
    'team_abbr': 'Team abbreviation',
    'season': 'Season year',
    'week': 'Week number (0 for season totals)',
    'avg_cushion': 'Average yards of cushion from CB at snap',
    'avg_separation': 'Average yards of separation at catch point',
    'receptions': 'Total receptions',
    'targets': 'Total targets',
    'catch_percentage': 'Reception rate',
    'yards': 'Total receiving yards',
    'avg_yac': 'Average yards after catch',
    'avg_expected_yac': 'Expected YAC based on situation',
    'avg_yac_above_expectation': 'YAC skill above expectation',
    'rec_touchdowns': 'Receiving touchdowns',
    'percent_share_of_intended_air_yards': 'Target share of team air yards',
}


def main():
    """Main ingestion script for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest NFL Next Gen Stats data")
    parser.add_argument('--seasons', nargs='+', type=int, default=None,
                        help="Seasons to ingest (default: 2016 to current)")
    parser.add_argument('--force-refresh', action='store_true',
                        help="Force re-download of all data")
    parser.add_argument('--stat-type', choices=['passing', 'rushing', 'receiving', 'all'],
                        default='all', help="Stat type to ingest")

    args = parser.parse_args()

    ingester = NGSIngester()

    if args.stat_type == 'all':
        data = ingester.ingest_all(args.seasons, args.force_refresh)
    else:
        if args.stat_type == 'passing':
            data = {'passing': ingester.get_passing(args.seasons)}
        elif args.stat_type == 'rushing':
            data = {'rushing': ingester.get_rushing(args.seasons)}
        else:
            data = {'receiving': ingester.get_receiving(args.seasons)}

    # Print summary
    print("\n=== NGS Data Summary ===")
    for stat_type, df in data.items():
        if not df.empty:
            print(f"\n{stat_type.upper()}:")
            print(f"  Records: {len(df):,}")
            print(f"  Seasons: {sorted(df['season'].unique())}")
            print(f"  Players: {df['player_gsis_id'].nunique() if 'player_gsis_id' in df.columns else 'N/A'}")
            print(f"  Columns: {list(df.columns)[:10]}...")
        else:
            print(f"\n{stat_type.upper()}: No data")


if __name__ == "__main__":
    main()
