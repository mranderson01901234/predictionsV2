"""
Advanced Game Features Generator

Generates game-level features from all advanced data sources:
- NGS (Next Gen Stats): Player tracking metrics
- FTN (Football Technology Network): Play charting data
- PFR (Pro Football Reference): Advanced box score stats

Combines all data sources into a unified feature set for game prediction.

Usage:
    from features.nfl.advanced_game_features import AdvancedGameFeatureGenerator
    generator = AdvancedGameFeatureGenerator()
    features = generator.generate_game_features(
        game_id="nfl_2024_10_KC_DEN",
        home_team="DEN",
        away_team="KC",
        season=2024,
        week=10
    )
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FTNFeatureExtractor:
    """
    Extract predictive features from FTN charting data.

    FTN provides play-by-play charting details like:
    - Play action usage
    - Blitz rates faced
    - Pressure rates
    - Catchable throw rates
    """

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize FTN feature extractor.

        Args:
            data_dir: Directory containing FTN cache
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data" / "nfl" / "raw" / "ftn"
        self.data_dir = Path(data_dir)
        self.ftn_df = None
        self._loaded = False

    def load_data(self, force_reload: bool = False) -> None:
        """Load FTN data from cache."""
        if self._loaded and not force_reload:
            return

        cache_path = self.data_dir / "ftn_data.parquet"
        if cache_path.exists():
            self.ftn_df = pd.read_parquet(cache_path)
            logger.info(f"Loaded FTN data: {len(self.ftn_df)} plays")
        else:
            logger.warning(f"No FTN data at {cache_path}")
            self.ftn_df = pd.DataFrame()

        self._loaded = True

    def compute_team_features(
        self,
        team: str,
        season: int,
        week: int,
        is_offense: bool = True
    ) -> Dict[str, float]:
        """
        Compute FTN-based features for a team.

        Args:
            team: Team abbreviation
            season: Season
            week: Week (uses data before this week)
            is_offense: True for offensive features, False for defensive

        Returns:
            Dict of feature name -> value
        """
        self.load_data()

        if self.ftn_df.empty:
            return {}

        # Filter to past data for this team
        if is_offense:
            team_col = 'posteam' if 'posteam' in self.ftn_df.columns else 'offense_team'
        else:
            team_col = 'defteam' if 'defteam' in self.ftn_df.columns else 'defense_team'

        if team_col not in self.ftn_df.columns:
            return {}

        mask = (
            (self.ftn_df[team_col] == team) &
            (
                (self.ftn_df['season'] < season) |
                ((self.ftn_df['season'] == season) & (self.ftn_df['week'] < week))
            )
        )
        team_plays = self.ftn_df[mask]

        if team_plays.empty or len(team_plays) < 50:
            return {}

        features = {}
        prefix = 'off_ftn' if is_offense else 'def_ftn'

        # Calculate rates for boolean columns
        bool_cols = {
            'is_play_action': 'play_action_rate',
            'is_screen': 'screen_rate',
            'is_rpo': 'rpo_rate',
            'is_blitz': 'blitz_rate',
            'is_qb_out_of_pocket': 'qb_out_of_pocket_rate',
            'is_catchable': 'catchable_rate',
            'is_contested': 'contested_catch_rate',
            'is_drop': 'drop_rate',
            'is_throw_away': 'throwaway_rate',
            'is_qb_hit': 'qb_hit_rate',
            'is_no_huddle': 'no_huddle_rate',
            'is_motion': 'motion_rate',
        }

        for col, feature_name in bool_cols.items():
            if col in team_plays.columns:
                # Handle both bool and int columns
                values = team_plays[col]
                if values.dtype == bool:
                    rate = values.mean()
                else:
                    rate = (values == 1).mean() if values.notna().any() else None

                if rate is not None and not np.isnan(rate):
                    features[f'{prefix}_{feature_name}'] = rate

        # Pass rushers (for defensive features)
        if not is_offense and 'n_pass_rushers' in team_plays.columns:
            values = team_plays['n_pass_rushers'].dropna()
            if len(values) > 0:
                features[f'{prefix}_avg_pass_rushers'] = values.mean()

        return features


class PFRFeatureExtractor:
    """
    Extract predictive features from PFR advanced stats.

    PFR provides advanced metrics like:
    - Pressure rates
    - On-target throw rates
    - Yards before/after contact
    """

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize PFR feature extractor.

        Args:
            data_dir: Directory containing PFR cache
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data" / "nfl" / "raw" / "pfr"
        self.data_dir = Path(data_dir)
        self.pass_df = None
        self.rush_df = None
        self.rec_df = None
        self._loaded = False

    def load_data(self, force_reload: bool = False) -> None:
        """Load PFR data from cache."""
        if self._loaded and not force_reload:
            return

        pass_path = self.data_dir / "pfr_weekly_pass.parquet"
        rush_path = self.data_dir / "pfr_weekly_rush.parquet"
        rec_path = self.data_dir / "pfr_weekly_rec.parquet"

        if pass_path.exists():
            self.pass_df = pd.read_parquet(pass_path)
            logger.info(f"Loaded PFR passing: {len(self.pass_df)} records")
        else:
            self.pass_df = pd.DataFrame()

        if rush_path.exists():
            self.rush_df = pd.read_parquet(rush_path)
            logger.info(f"Loaded PFR rushing: {len(self.rush_df)} records")
        else:
            self.rush_df = pd.DataFrame()

        if rec_path.exists():
            self.rec_df = pd.read_parquet(rec_path)
            logger.info(f"Loaded PFR receiving: {len(self.rec_df)} records")
        else:
            self.rec_df = pd.DataFrame()

        self._loaded = True

    def compute_team_qb_features(
        self,
        team: str,
        season: int,
        week: int
    ) -> Dict[str, float]:
        """
        Compute PFR QB features for a team.

        Args:
            team: Team abbreviation
            season: Season
            week: Week

        Returns:
            Dict of QB features
        """
        self.load_data()

        if self.pass_df.empty:
            return {}

        team_col = 'team' if 'team' in self.pass_df.columns else 'team_abbr'
        if team_col not in self.pass_df.columns:
            return {}

        # Filter to team's past data
        mask = (
            (self.pass_df[team_col] == team) &
            (
                (self.pass_df['season'] < season) |
                ((self.pass_df['season'] == season) & (self.pass_df['week'] < week))
            )
        )
        team_data = self.pass_df[mask]

        if team_data.empty:
            return {}

        # Get primary QB (most attempts)
        attempts_col = 'attempts' if 'attempts' in team_data.columns else None
        player_col = 'pfr_player_id' if 'pfr_player_id' in team_data.columns else 'player'

        if attempts_col and player_col in team_data.columns:
            recent = team_data.sort_values(['season', 'week']).tail(4)
            primary_qb = recent.groupby(player_col)[attempts_col].sum().idxmax()
            qb_data = team_data[team_data[player_col] == primary_qb].tail(8)
        else:
            qb_data = team_data.sort_values(['season', 'week']).tail(8)

        if qb_data.empty:
            return {}

        features = {}

        # Pressure-related metrics
        pressure_cols = ['times_pressured', 'times_hurried', 'times_hit', 'times_blitzed']
        dropback_col = 'dropbacks' if 'dropbacks' in qb_data.columns else 'attempts'

        if dropback_col in qb_data.columns:
            dropbacks = qb_data[dropback_col].sum()
            if dropbacks > 0:
                for col in pressure_cols:
                    if col in qb_data.columns:
                        rate = qb_data[col].sum() / dropbacks
                        features[f'qb_pfr_{col.replace("times_", "")}_rate'] = rate

        # Accuracy metrics
        if 'on_target_throws' in qb_data.columns:
            attempts = qb_data['attempts'].sum() if 'attempts' in qb_data.columns else len(qb_data)
            if attempts > 0:
                features['qb_pfr_on_target_rate'] = qb_data['on_target_throws'].sum() / attempts

        if 'bad_throws' in qb_data.columns:
            attempts = qb_data['attempts'].sum() if 'attempts' in qb_data.columns else len(qb_data)
            if attempts > 0:
                features['qb_pfr_bad_throw_rate'] = qb_data['bad_throws'].sum() / attempts

        # Pocket time
        if 'pocket_time' in qb_data.columns:
            values = qb_data['pocket_time'].dropna()
            if len(values) > 0:
                features['qb_pfr_pocket_time'] = values.mean()

        return features

    def compute_team_rb_features(
        self,
        team: str,
        season: int,
        week: int
    ) -> Dict[str, float]:
        """
        Compute PFR RB features for a team.

        Args:
            team: Team abbreviation
            season: Season
            week: Week

        Returns:
            Dict of RB features
        """
        self.load_data()

        if self.rush_df.empty:
            return {}

        team_col = 'team' if 'team' in self.rush_df.columns else 'team_abbr'
        if team_col not in self.rush_df.columns:
            return {}

        mask = (
            (self.rush_df[team_col] == team) &
            (
                (self.rush_df['season'] < season) |
                ((self.rush_df['season'] == season) & (self.rush_df['week'] < week))
            )
        )
        team_data = self.rush_df[mask].sort_values(['season', 'week']).tail(16)

        if team_data.empty:
            return {}

        features = {}

        # Yards before/after contact
        carries = team_data['carries'].sum() if 'carries' in team_data.columns else len(team_data)
        if carries > 0:
            if 'ybc' in team_data.columns:
                features['rb_pfr_ybc_per_carry'] = team_data['ybc'].sum() / carries
            if 'yac' in team_data.columns:
                features['rb_pfr_yac_per_carry'] = team_data['yac'].sum() / carries
            if 'broken_tackles' in team_data.columns:
                features['rb_pfr_broken_tackle_rate'] = team_data['broken_tackles'].sum() / carries

        return features


class AdvancedGameFeatureGenerator:
    """
    Generate complete advanced feature set for game prediction.

    Combines all data sources:
    - NGS features (CPOE, time to throw, RYOE, separation)
    - FTN features (play action rate, blitz rate, pressure rate)
    - PFR features (pressure rate, bad throw rate, YBC/YAC)
    """

    def __init__(self):
        """Initialize the generator with all extractors."""
        from features.nfl.ngs_features import NGSFeatureExtractor, NGSTeamAggregator

        self.ngs_extractor = NGSFeatureExtractor()
        self.ngs_aggregator = NGSTeamAggregator(self.ngs_extractor)
        self.ftn_extractor = FTNFeatureExtractor()
        self.pfr_extractor = PFRFeatureExtractor()

    def generate_game_features(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        season: int,
        week: int
    ) -> Dict[str, any]:
        """
        Generate all advanced features for a single game.

        Args:
            game_id: Unique game identifier
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: Season year
            week: Week number

        Returns:
            Dict of feature_name -> feature_value
        """
        features = {
            'game_id': game_id,
            'season': season,
            'week': week,
            'home_team': home_team,
            'away_team': away_team,
        }

        # NGS matchup features (includes differential)
        ngs_features = self.ngs_aggregator.compute_matchup_features(
            home_team, away_team, season, week
        )
        features.update(ngs_features)

        # FTN features for each team
        home_ftn_off = self.ftn_extractor.compute_team_features(
            home_team, season, week, is_offense=True
        )
        home_ftn_def = self.ftn_extractor.compute_team_features(
            home_team, season, week, is_offense=False
        )
        away_ftn_off = self.ftn_extractor.compute_team_features(
            away_team, season, week, is_offense=True
        )
        away_ftn_def = self.ftn_extractor.compute_team_features(
            away_team, season, week, is_offense=False
        )

        for k, v in home_ftn_off.items():
            features[f'home_{k}'] = v
        for k, v in home_ftn_def.items():
            features[f'home_{k}'] = v
        for k, v in away_ftn_off.items():
            features[f'away_{k}'] = v
        for k, v in away_ftn_def.items():
            features[f'away_{k}'] = v

        # PFR features
        home_pfr_qb = self.pfr_extractor.compute_team_qb_features(home_team, season, week)
        home_pfr_rb = self.pfr_extractor.compute_team_rb_features(home_team, season, week)
        away_pfr_qb = self.pfr_extractor.compute_team_qb_features(away_team, season, week)
        away_pfr_rb = self.pfr_extractor.compute_team_rb_features(away_team, season, week)

        for k, v in home_pfr_qb.items():
            features[f'home_{k}'] = v
        for k, v in home_pfr_rb.items():
            features[f'home_{k}'] = v
        for k, v in away_pfr_qb.items():
            features[f'away_{k}'] = v
        for k, v in away_pfr_rb.items():
            features[f'away_{k}'] = v

        return features

    def generate_all_game_features(
        self,
        schedule_df: pd.DataFrame,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Generate features for all games in a schedule.

        Args:
            schedule_df: DataFrame with columns:
                - game_id
                - home_team
                - away_team
                - season
                - week
            show_progress: Print progress updates

        Returns:
            DataFrame with one row per game and all features
        """
        all_features = []
        total = len(schedule_df)

        for idx, row in schedule_df.iterrows():
            if show_progress and idx % 50 == 0:
                logger.info(f"Processing game {idx + 1}/{total}")

            try:
                game_features = self.generate_game_features(
                    game_id=row['game_id'],
                    home_team=row['home_team'],
                    away_team=row['away_team'],
                    season=row['season'],
                    week=row['week'],
                )
                all_features.append(game_features)

            except Exception as e:
                logger.error(f"Error generating features for {row.get('game_id', idx)}: {e}")
                continue

        result_df = pd.DataFrame(all_features)
        logger.info(f"Generated features for {len(result_df)} games")

        return result_df


def merge_with_existing_features(
    existing_path: str,
    advanced_features_df: pd.DataFrame,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Merge advanced features with existing feature pipeline.

    Args:
        existing_path: Path to existing features parquet
        advanced_features_df: DataFrame with advanced features
        output_path: Path to save merged features (optional)

    Returns:
        Merged DataFrame
    """
    existing_df = pd.read_parquet(existing_path)
    logger.info(f"Loaded existing features: {existing_df.shape}")

    # Merge on game_id
    merged = existing_df.merge(
        advanced_features_df,
        on=['game_id'],
        how='left',
        suffixes=('', '_adv')
    )

    # Handle duplicate columns
    for col in merged.columns:
        if col.endswith('_adv'):
            base_col = col[:-4]
            # Keep advanced version if existing is null
            if base_col in merged.columns:
                merged[base_col] = merged[base_col].fillna(merged[col])
            merged = merged.drop(columns=[col])

    logger.info(f"Merged features: {merged.shape}")
    logger.info(f"New columns: {len(merged.columns) - len(existing_df.columns)}")

    if output_path:
        merged.to_parquet(output_path, index=False)
        logger.info(f"Saved to {output_path}")

    return merged


def main():
    """Generate advanced features for all games."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate advanced game features")
    parser.add_argument('--existing-features', type=str, required=True,
                        help="Path to existing features parquet")
    parser.add_argument('--output', type=str, default=None,
                        help="Output path (default: append _ngs to existing)")
    parser.add_argument('--seasons', nargs='+', type=int, default=None,
                        help="Seasons to process (default: all)")

    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        existing_path = Path(args.existing_features)
        output_path = existing_path.parent / f"{existing_path.stem}_ngs.parquet"
    else:
        output_path = Path(args.output)

    # Load existing features to get schedule
    existing_df = pd.read_parquet(args.existing_features)

    # Filter seasons if specified
    if args.seasons:
        schedule = existing_df[existing_df['season'].isin(args.seasons)][
            ['game_id', 'home_team', 'away_team', 'season', 'week']
        ].drop_duplicates()
    else:
        schedule = existing_df[
            ['game_id', 'home_team', 'away_team', 'season', 'week']
        ].drop_duplicates()

    logger.info(f"Processing {len(schedule)} games")

    # Generate advanced features
    generator = AdvancedGameFeatureGenerator()
    advanced_df = generator.generate_all_game_features(schedule)

    # Merge with existing
    merged_df = merge_with_existing_features(
        args.existing_features,
        advanced_df,
        str(output_path)
    )

    print(f"\n=== Feature Generation Complete ===")
    print(f"Games processed: {len(schedule)}")
    print(f"Total features: {len(merged_df.columns)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
