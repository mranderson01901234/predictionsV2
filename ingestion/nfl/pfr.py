"""
Pro Football Reference (PFR) Advanced Stats Ingestion Module

Fetches PFR advanced stats via nflverse.
PFR provides advanced metrics not in standard box scores.

Available historically, but advanced gamelog data is 2018+.

Key metrics:
- Pressure rate (% of dropbacks under pressure)
- On-target throw rate
- Bad throw rate
- Scramble rate
- Time to throw (alternative to NGS)

Usage:
    from ingestion.nfl.pfr import PFRIngester
    ingester = PFRIngester()
    data = ingester.ingest_weekly(seasons=[2018, 2019, 2020, 2021, 2022, 2023, 2024])
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
    logger.info("Using nflreadpy for PFR data")
except ImportError:
    try:
        import nfl_data_py as nfl
        USING_NFLREADPY = False
        logger.info("Using nfl_data_py for PFR data")
    except ImportError:
        raise ImportError(
            "Either nflreadpy or nfl_data_py is required. "
            "Install with: pip install nflreadpy or pip install nfl-data-py"
        )


class PFRIngester:
    """
    Download and cache Pro Football Reference advanced stats.

    PFR provides advanced metrics that go beyond standard box scores,
    including pressure rates, throw accuracy, and detailed rushing stats.
    """

    # PFR advanced data available from 2018
    MIN_SEASON = 2018

    # Stat types available
    STAT_TYPES = ['pass', 'rush', 'rec']

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the PFR ingester.

        Args:
            cache_dir: Directory to cache data. Defaults to data/nfl/raw/pfr
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "data" / "nfl" / "raw" / "pfr"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def ingest_weekly(
        self,
        seasons: Optional[List[int]] = None,
        force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Ingest weekly PFR advanced stats.

        Args:
            seasons: List of seasons (default: 2018 to current)
            force_refresh: Re-download even if cached

        Returns:
            Dict mapping stat_type to DataFrame
        """
        if seasons is None:
            import datetime
            current_year = datetime.datetime.now().year
            seasons = list(range(self.MIN_SEASON, current_year + 1))

        seasons = [s for s in seasons if s >= self.MIN_SEASON]

        if not seasons:
            logger.warning(f"No valid seasons for PFR data (requires {self.MIN_SEASON}+)")
            return {}

        all_data = {}

        for stat_type in self.STAT_TYPES:
            logger.info(f"Ingesting PFR weekly {stat_type} data for seasons {seasons}")

            cache_path = self.cache_dir / f"pfr_weekly_{stat_type}.parquet"

            if cache_path.exists() and not force_refresh:
                logger.info(f"  Loading from cache: {cache_path}")
                df = pd.read_parquet(cache_path)

                # Check for missing seasons
                cached_seasons = df['season'].unique().tolist() if 'season' in df.columns else []
                missing_seasons = [s for s in seasons if s not in cached_seasons]

                if missing_seasons:
                    logger.info(f"  Fetching missing seasons: {missing_seasons}")
                    new_df = self._fetch_pfr_weekly(stat_type, missing_seasons)
                    if not new_df.empty:
                        df = pd.concat([df, new_df], ignore_index=True)
                        # Deduplicate
                        dedup_cols = ['season', 'week', 'pfr_player_id'] if 'pfr_player_id' in df.columns else ['season', 'week']
                        df = df.drop_duplicates(subset=dedup_cols, keep='last')
                        self._save_cache(df, cache_path)
            else:
                df = self._fetch_pfr_weekly(stat_type, seasons)
                if not df.empty:
                    self._save_cache(df, cache_path)

            all_data[stat_type] = df
            if not df.empty:
                logger.info(f"  {stat_type}: {len(df)} records")
            else:
                logger.warning(f"  {stat_type}: No data fetched")

        return all_data

    def ingest_seasonal(
        self,
        seasons: Optional[List[int]] = None,
        force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Ingest seasonal PFR advanced stats (season totals).

        Args:
            seasons: List of seasons (default: 2018 to current)
            force_refresh: Re-download even if cached

        Returns:
            Dict mapping stat_type to DataFrame
        """
        if seasons is None:
            import datetime
            current_year = datetime.datetime.now().year
            seasons = list(range(self.MIN_SEASON, current_year + 1))

        seasons = [s for s in seasons if s >= self.MIN_SEASON]

        if not seasons:
            logger.warning(f"No valid seasons for PFR data (requires {self.MIN_SEASON}+)")
            return {}

        all_data = {}

        for stat_type in self.STAT_TYPES:
            logger.info(f"Ingesting PFR seasonal {stat_type} data for seasons {seasons}")

            cache_path = self.cache_dir / f"pfr_seasonal_{stat_type}.parquet"

            if cache_path.exists() and not force_refresh:
                logger.info(f"  Loading from cache: {cache_path}")
                df = pd.read_parquet(cache_path)

                # Check for missing seasons
                cached_seasons = df['season'].unique().tolist() if 'season' in df.columns else []
                missing_seasons = [s for s in seasons if s not in cached_seasons]

                if missing_seasons:
                    logger.info(f"  Fetching missing seasons: {missing_seasons}")
                    new_df = self._fetch_pfr_seasonal(stat_type, missing_seasons)
                    if not new_df.empty:
                        df = pd.concat([df, new_df], ignore_index=True)
                        df = df.drop_duplicates(subset=['season', 'pfr_player_id'] if 'pfr_player_id' in df.columns else ['season'], keep='last')
                        self._save_cache(df, cache_path)
            else:
                df = self._fetch_pfr_seasonal(stat_type, seasons)
                if not df.empty:
                    self._save_cache(df, cache_path)

            all_data[stat_type] = df

        return all_data

    def _fetch_pfr_weekly(self, stat_type: str, seasons: List[int]) -> pd.DataFrame:
        """
        Fetch weekly PFR data from nflverse.

        Args:
            stat_type: One of 'pass', 'rush', 'rec'
            seasons: List of seasons to fetch

        Returns:
            DataFrame with weekly PFR data
        """
        try:
            if USING_NFLREADPY:
                df = nfl.load_pfr_advstats(seasons=seasons, stat_type=stat_type)
                if hasattr(df, 'to_pandas'):
                    df = df.to_pandas()
            else:
                df = nfl.import_weekly_pfr(stat_type, seasons)

            logger.info(f"  Fetched {len(df)} {stat_type} records")
            return df

        except Exception as e:
            logger.error(f"Error fetching PFR weekly {stat_type}: {e}")
            return pd.DataFrame()

    def _fetch_pfr_seasonal(self, stat_type: str, seasons: List[int]) -> pd.DataFrame:
        """
        Fetch seasonal PFR data from nflverse.

        Args:
            stat_type: One of 'pass', 'rush', 'rec'
            seasons: List of seasons to fetch

        Returns:
            DataFrame with seasonal PFR data
        """
        try:
            # Note: nflreadpy may not have seasonal PFR
            if USING_NFLREADPY:
                # Try to get seasonal data - may need weekly and aggregate
                df = nfl.import_seasonal_pfr(stat_type, seasons)
            else:
                df = nfl.import_seasonal_pfr(stat_type, seasons)

            logger.info(f"  Fetched {len(df)} seasonal {stat_type} records")
            return df

        except Exception as e:
            logger.error(f"Error fetching PFR seasonal {stat_type}: {e}")
            return pd.DataFrame()

    def _save_cache(self, df: pd.DataFrame, path: Path) -> None:
        """Save DataFrame to parquet cache."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info(f"  Cached to {path}")

    def get_passing(self, seasons: Optional[List[int]] = None) -> pd.DataFrame:
        """Get passing PFR data."""
        return self.ingest_weekly(seasons).get('pass', pd.DataFrame())

    def get_rushing(self, seasons: Optional[List[int]] = None) -> pd.DataFrame:
        """Get rushing PFR data."""
        return self.ingest_weekly(seasons).get('rush', pd.DataFrame())

    def get_receiving(self, seasons: Optional[List[int]] = None) -> pd.DataFrame:
        """Get receiving PFR data."""
        return self.ingest_weekly(seasons).get('rec', pd.DataFrame())

    def get_summary(self) -> Dict[str, Dict]:
        """
        Get summary of cached PFR data.

        Returns:
            Dict with stats about each data type
        """
        summary = {}

        for stat_type in self.STAT_TYPES:
            # Weekly data
            weekly_path = self.cache_dir / f"pfr_weekly_{stat_type}.parquet"
            if weekly_path.exists():
                df = pd.read_parquet(weekly_path)
                summary[f'weekly_{stat_type}'] = {
                    'records': len(df),
                    'seasons': sorted(df['season'].unique().tolist()) if 'season' in df.columns else [],
                    'columns': list(df.columns),
                }
            else:
                summary[f'weekly_{stat_type}'] = {'records': 0, 'seasons': [], 'columns': []}

            # Seasonal data
            seasonal_path = self.cache_dir / f"pfr_seasonal_{stat_type}.parquet"
            if seasonal_path.exists():
                df = pd.read_parquet(seasonal_path)
                summary[f'seasonal_{stat_type}'] = {
                    'records': len(df),
                    'seasons': sorted(df['season'].unique().tolist()) if 'season' in df.columns else [],
                    'columns': list(df.columns),
                }
            else:
                summary[f'seasonal_{stat_type}'] = {'records': 0, 'seasons': [], 'columns': []}

        return summary


# PFR Passing Columns Reference
PFR_PASSING_COLUMNS = {
    'pfr_player_id': 'PFR unique player identifier',
    'player': 'Player name',
    'team': 'Team abbreviation',
    'season': 'Season year',
    'week': 'Week number',
    'game_id': 'Game identifier',

    # Pressure stats
    'times_pressured': 'Times QB was pressured',
    'times_hurried': 'Times QB was hurried (pressure without contact)',
    'times_hit': 'Times QB was hit after throw',
    'times_blitzed': 'Times defense sent extra rushers',
    'times_scrambled': 'Times QB scrambled',

    # Accuracy stats
    'on_target_throws': 'Throws that were accurate',
    'bad_throws': 'Inaccurate throws',
    'batted_balls': 'Passes batted at line of scrimmage',
    'throw_aways': 'Intentional throwaways',
    'drops': 'Passes dropped by receivers',

    # Timing
    'pocket_time': 'Average time in pocket before throw',

    # Derived metrics (compute these)
    # 'pressure_rate': times_pressured / dropbacks
    # 'on_target_rate': on_target_throws / attempts
    # 'bad_throw_rate': bad_throws / attempts
    # 'drop_rate': drops / (completions + drops)
}

PFR_RUSHING_COLUMNS = {
    'pfr_player_id': 'PFR unique player identifier',
    'player': 'Player name',
    'team': 'Team abbreviation',
    'season': 'Season year',
    'week': 'Week number',

    # Rushing stats
    'carries': 'Rush attempts',
    'yards': 'Rushing yards',
    'tds': 'Rushing touchdowns',
    'first_downs': 'First downs via rush',
    'ybc': 'Yards before contact',
    'yac': 'Yards after contact',
    'broken_tackles': 'Broken tackles',

    # Derived metrics
    # 'ybc_per_carry': ybc / carries
    # 'yac_per_carry': yac / carries
    # 'broken_tackle_rate': broken_tackles / carries
}

PFR_RECEIVING_COLUMNS = {
    'pfr_player_id': 'PFR unique player identifier',
    'player': 'Player name',
    'team': 'Team abbreviation',
    'season': 'Season year',
    'week': 'Week number',

    # Receiving stats
    'targets': 'Times targeted',
    'receptions': 'Catches',
    'yards': 'Receiving yards',
    'tds': 'Receiving touchdowns',
    'first_downs': 'First downs via reception',
    'ybc': 'Yards before catch (air yards)',
    'yac': 'Yards after catch',
    'broken_tackles': 'Broken tackles after catch',
    'drops': 'Dropped passes',
    'int_when_targeted': 'Interceptions on passes to this receiver',

    # Derived metrics
    # 'catch_rate': receptions / targets
    # 'drop_rate': drops / (receptions + drops)
    # 'yac_per_catch': yac / receptions
}


def main():
    """Main ingestion script for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest PFR advanced stats")
    parser.add_argument('--seasons', nargs='+', type=int, default=None,
                        help="Seasons to ingest (default: 2018 to current)")
    parser.add_argument('--force-refresh', action='store_true',
                        help="Force re-download of all data")
    parser.add_argument('--weekly-only', action='store_true',
                        help="Only fetch weekly data (skip seasonal)")
    parser.add_argument('--seasonal-only', action='store_true',
                        help="Only fetch seasonal data (skip weekly)")
    parser.add_argument('--stat-type', choices=['pass', 'rush', 'rec', 'all'],
                        default='all', help="Stat type to ingest")

    args = parser.parse_args()

    ingester = PFRIngester()

    # Ingest data
    weekly_data = {}
    seasonal_data = {}

    if not args.seasonal_only:
        weekly_data = ingester.ingest_weekly(args.seasons, args.force_refresh)

    if not args.weekly_only:
        seasonal_data = ingester.ingest_seasonal(args.seasons, args.force_refresh)

    # Print summary
    print("\n=== PFR Data Summary ===")

    if weekly_data:
        print("\n--- Weekly Data ---")
        for stat_type, df in weekly_data.items():
            if not df.empty:
                print(f"\n{stat_type.upper()}:")
                print(f"  Records: {len(df):,}")
                print(f"  Seasons: {sorted(df['season'].unique()) if 'season' in df.columns else 'N/A'}")
                print(f"  Columns: {list(df.columns)[:10]}...")

    if seasonal_data:
        print("\n--- Seasonal Data ---")
        for stat_type, df in seasonal_data.items():
            if not df.empty:
                print(f"\n{stat_type.upper()}:")
                print(f"  Records: {len(df):,}")
                print(f"  Seasons: {sorted(df['season'].unique()) if 'season' in df.columns else 'N/A'}")


if __name__ == "__main__":
    main()
