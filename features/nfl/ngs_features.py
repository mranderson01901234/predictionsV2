"""
NFL Next Gen Stats Feature Extraction Module

Extracts predictive features from NGS data for game prediction.

Key features by position:
- QB: CPOE, time_to_throw, aggressiveness, air_yards
- RB: RYOE, efficiency, time_to_LOS
- WR/TE: separation, cushion, YAC above expected

CRITICAL: All features use only data BEFORE the game being predicted.
For a game in Week 5, we only use data from Weeks 1-4.

Usage:
    from features.nfl.ngs_features import NGSFeatureExtractor
    extractor = NGSFeatureExtractor()
    qb_features = extractor.compute_qb_features(player_id, season=2024, week=5)
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NGSFeatureExtractor:
    """
    Extract predictive features from NGS data.

    CRITICAL: All features use only past data to prevent leakage.
    For a game in Week 5, we only use data from Weeks 1-4.
    """

    # Rolling window sizes (last N games)
    WINDOWS = [3, 5, 8]

    # Key metrics by position
    QB_METRICS = [
        'avg_time_to_throw',
        'avg_completed_air_yards',
        'avg_intended_air_yards',
        'avg_air_yards_differential',
        'aggressiveness',
        'completion_percentage',
        'expected_completion_percentage',
        'completion_percentage_above_expectation',  # CPOE - most important
        'passer_rating',
        'avg_air_yards_to_sticks',
    ]

    RB_METRICS = [
        'efficiency',  # Lower = more north-south
        'percent_attempts_gte_eight_defenders',  # Stacked boxes faced
        'avg_time_to_los',  # Time behind LOS
        'rush_yards',
        'rush_attempts',
        'avg_rush_yards',
        'expected_rush_yards',
        'rush_yards_over_expected',
        'rush_yards_over_expected_per_att',  # RYOE per attempt - key metric
    ]

    WR_METRICS = [
        'avg_cushion',  # Press vs off coverage
        'avg_separation',  # Route running ability
        'receptions',
        'targets',
        'catch_percentage',
        'yards',
        'avg_yac',
        'avg_expected_yac',
        'avg_yac_above_expectation',  # YAC skill
        'percent_share_of_intended_air_yards',  # Target share
    ]

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the NGS feature extractor.

        Args:
            data_dir: Directory containing NGS data cache. Defaults to data/nfl/raw/ngs
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data" / "nfl" / "raw" / "ngs"
        self.data_dir = Path(data_dir)

        self.passing_df = None
        self.rushing_df = None
        self.receiving_df = None
        self._data_loaded = False

    def load_data(self, force_reload: bool = False) -> None:
        """
        Load cached NGS data.

        Args:
            force_reload: Force reload even if already loaded
        """
        if self._data_loaded and not force_reload:
            return

        passing_path = self.data_dir / "passing" / "ngs_data.parquet"
        rushing_path = self.data_dir / "rushing" / "ngs_data.parquet"
        receiving_path = self.data_dir / "receiving" / "ngs_data.parquet"

        if passing_path.exists():
            self.passing_df = pd.read_parquet(passing_path)
            logger.info(f"Loaded passing NGS: {len(self.passing_df)} records")
        else:
            logger.warning(f"No passing NGS data at {passing_path}")
            self.passing_df = pd.DataFrame()

        if rushing_path.exists():
            self.rushing_df = pd.read_parquet(rushing_path)
            logger.info(f"Loaded rushing NGS: {len(self.rushing_df)} records")
        else:
            logger.warning(f"No rushing NGS data at {rushing_path}")
            self.rushing_df = pd.DataFrame()

        if receiving_path.exists():
            self.receiving_df = pd.read_parquet(receiving_path)
            logger.info(f"Loaded receiving NGS: {len(self.receiving_df)} records")
        else:
            logger.warning(f"No receiving NGS data at {receiving_path}")
            self.receiving_df = pd.DataFrame()

        self._data_loaded = True

    def _filter_past_data(
        self,
        df: pd.DataFrame,
        player_id: str,
        season: int,
        week: int,
        player_id_col: str = 'player_gsis_id'
    ) -> pd.DataFrame:
        """
        Filter dataframe to only include past data for a player.

        CRITICAL: Only uses data from BEFORE the specified game.
        - Previous seasons: all weeks
        - Current season: weeks 1 through week-1

        Args:
            df: Source dataframe
            player_id: Player GSIS ID
            season: Current game season
            week: Current game week
            player_id_col: Column name for player ID

        Returns:
            Filtered DataFrame with only past data
        """
        if df.empty or player_id_col not in df.columns:
            return pd.DataFrame()

        mask = (
            (df[player_id_col] == player_id) &
            (
                (df['season'] < season) |
                (
                    (df['season'] == season) &
                    (df['week'] < week) &
                    (df['week'] > 0)  # Exclude season totals (week=0)
                )
            )
        )

        return df[mask].copy()

    def compute_qb_features(
        self,
        player_id: str,
        season: int,
        week: int,
        include_career: bool = True
    ) -> Dict[str, float]:
        """
        Compute QB features for a specific game.

        Uses only data from BEFORE this game (seasons < this season,
        or same season but week < this week).

        Args:
            player_id: Player GSIS ID
            season: Season of game to predict
            week: Week of game to predict
            include_career: Include career baseline features

        Returns:
            Dict of feature name -> value
        """
        self.load_data()

        if self.passing_df.empty:
            return {}

        # Filter to only this player's past data
        df = self._filter_past_data(self.passing_df, player_id, season, week)

        if df.empty:
            return {}

        # Sort by date (season, week)
        df = df.sort_values(['season', 'week'])

        features = {}

        # Rolling window features
        for window in self.WINDOWS:
            recent = df.tail(window)

            if len(recent) < max(1, window // 2):
                # Not enough data for this window
                continue

            for metric in self.QB_METRICS:
                if metric not in recent.columns:
                    continue

                values = recent[metric].dropna()
                if len(values) == 0:
                    continue

                # Mean
                features[f'qb_ngs_{metric}_L{window}'] = values.mean()

                # Std (for key metrics only)
                if metric in ['completion_percentage_above_expectation', 'avg_time_to_throw', 'aggressiveness']:
                    if len(values) > 1:
                        features[f'qb_ngs_{metric}_std_L{window}'] = values.std()

        # Career baseline features
        if include_career and len(df) >= 3:
            for metric in self.QB_METRICS:
                if metric not in df.columns:
                    continue

                values = df[metric].dropna()
                if len(values) > 0:
                    features[f'qb_ngs_{metric}_career'] = values.mean()

                    if len(values) > 3:
                        features[f'qb_ngs_{metric}_career_std'] = values.std()

        # Season-to-date features (current season only, before this week)
        current_season = df[df['season'] == season]
        if not current_season.empty:
            for metric in self.QB_METRICS:
                if metric not in current_season.columns:
                    continue

                values = current_season[metric].dropna()
                if len(values) > 0:
                    features[f'qb_ngs_{metric}_season'] = values.mean()

        # CPOE deviation from career (key feature)
        cpoe_season_key = 'qb_ngs_completion_percentage_above_expectation_season'
        cpoe_career_key = 'qb_ngs_completion_percentage_above_expectation_career'
        if cpoe_season_key in features and cpoe_career_key in features:
            features['qb_ngs_cpoe_vs_career'] = (
                features[cpoe_season_key] - features[cpoe_career_key]
            )

        # Trend features (is CPOE improving or declining?)
        if len(df) >= 4:
            cpoe_col = 'completion_percentage_above_expectation'
            if cpoe_col in df.columns:
                recent_4 = df.tail(4)[cpoe_col].dropna()
                if len(recent_4) >= 4:
                    # Simple trend: compare recent 2 vs prior 2
                    recent_2 = recent_4.tail(2).mean()
                    prior_2 = recent_4.head(2).mean()
                    features['qb_ngs_cpoe_trend'] = recent_2 - prior_2

        # Aggressiveness + CPOE interaction (elite indicator)
        agg_key = 'qb_ngs_aggressiveness_L5'
        cpoe_l5_key = 'qb_ngs_completion_percentage_above_expectation_L5'
        if agg_key in features and cpoe_l5_key in features:
            # High aggressiveness + high CPOE = elite
            features['qb_ngs_aggressive_cpoe_L5'] = (
                features[agg_key] * features[cpoe_l5_key]
            )

        return features

    def compute_rb_features(
        self,
        player_id: str,
        season: int,
        week: int
    ) -> Dict[str, float]:
        """
        Compute RB features for a specific game.

        Args:
            player_id: Player GSIS ID
            season: Season of game to predict
            week: Week of game to predict

        Returns:
            Dict of feature name -> value
        """
        self.load_data()

        if self.rushing_df.empty:
            return {}

        df = self._filter_past_data(self.rushing_df, player_id, season, week)

        if df.empty:
            return {}

        df = df.sort_values(['season', 'week'])

        features = {}

        for window in self.WINDOWS:
            recent = df.tail(window)

            if len(recent) < max(1, window // 2):
                continue

            for metric in self.RB_METRICS:
                if metric not in recent.columns:
                    continue

                values = recent[metric].dropna()
                if len(values) == 0:
                    continue

                features[f'rb_ngs_{metric}_L{window}'] = values.mean()

        # RYOE per attempt is the key RB skill metric - make it easily accessible
        ryoe_key = 'rb_ngs_rush_yards_over_expected_per_att_L5'
        if ryoe_key in features:
            features['rb_ngs_ryoe_per_att'] = features[ryoe_key]

        # Career baseline
        if len(df) >= 3:
            for metric in ['rush_yards_over_expected_per_att', 'efficiency']:
                if metric in df.columns:
                    values = df[metric].dropna()
                    if len(values) > 0:
                        features[f'rb_ngs_{metric}_career'] = values.mean()

        return features

    def compute_wr_features(
        self,
        player_id: str,
        season: int,
        week: int
    ) -> Dict[str, float]:
        """
        Compute WR/TE features for a specific game.

        Args:
            player_id: Player GSIS ID
            season: Season of game to predict
            week: Week of game to predict

        Returns:
            Dict of feature name -> value
        """
        self.load_data()

        if self.receiving_df.empty:
            return {}

        df = self._filter_past_data(self.receiving_df, player_id, season, week)

        if df.empty:
            return {}

        df = df.sort_values(['season', 'week'])

        features = {}

        for window in self.WINDOWS:
            recent = df.tail(window)

            if len(recent) < max(1, window // 2):
                continue

            for metric in self.WR_METRICS:
                if metric not in recent.columns:
                    continue

                values = recent[metric].dropna()
                if len(values) == 0:
                    continue

                features[f'wr_ngs_{metric}_L{window}'] = values.mean()

        # Career baseline for key metrics
        if len(df) >= 3:
            for metric in ['avg_separation', 'avg_yac_above_expectation']:
                if metric in df.columns:
                    values = df[metric].dropna()
                    if len(values) > 0:
                        features[f'wr_ngs_{metric}_career'] = values.mean()

        return features

    def compute_team_qb_features(
        self,
        team: str,
        season: int,
        week: int,
    ) -> Dict[str, float]:
        """
        Compute team's QB features based on expected starter.

        Identifies the primary QB from recent games and computes their features.

        Args:
            team: Team abbreviation (KC, BUF, etc.)
            season: Season
            week: Week

        Returns:
            Dict of QB features for the team
        """
        self.load_data()

        if self.passing_df.empty:
            return {}

        # Find the team's QBs from past data
        mask = (
            (self.passing_df['team_abbr'] == team) &
            (
                (self.passing_df['season'] < season) |
                (
                    (self.passing_df['season'] == season) &
                    (self.passing_df['week'] < week) &
                    (self.passing_df['week'] > 0)
                )
            )
        )
        team_qbs = self.passing_df[mask]

        if team_qbs.empty:
            return {}

        # Get QB with most attempts in recent games (last 4 weeks)
        recent = team_qbs.sort_values(['season', 'week']).tail(4)
        attempts_col = 'attempts' if 'attempts' in recent.columns else None

        if attempts_col:
            primary_qb = recent.groupby('player_gsis_id')[attempts_col].sum().idxmax()
        else:
            # Fall back to most recent games
            primary_qb = recent.tail(1)['player_gsis_id'].values[0]

        return self.compute_qb_features(primary_qb, season, week)

    def compute_team_rb_features(
        self,
        team: str,
        season: int,
        week: int,
        top_n: int = 2
    ) -> Dict[str, float]:
        """
        Compute aggregated RB features for a team.

        Args:
            team: Team abbreviation
            season: Season
            week: Week
            top_n: Number of top RBs to include

        Returns:
            Dict of aggregated RB features
        """
        self.load_data()

        if self.rushing_df.empty:
            return {}

        # Find team's RBs from past data
        mask = (
            (self.rushing_df['team_abbr'] == team) &
            (
                (self.rushing_df['season'] < season) |
                (
                    (self.rushing_df['season'] == season) &
                    (self.rushing_df['week'] < week) &
                    (self.rushing_df['week'] > 0)
                )
            )
        )
        team_rbs = self.rushing_df[mask]

        if team_rbs.empty:
            return {}

        # Get top RBs by attempts
        recent = team_rbs.sort_values(['season', 'week']).groupby('player_gsis_id').tail(4)
        attempts_col = 'rush_attempts' if 'rush_attempts' in recent.columns else None

        if attempts_col:
            top_rbs = (
                recent.groupby('player_gsis_id')[attempts_col]
                .sum()
                .nlargest(top_n)
                .index.tolist()
            )
        else:
            top_rbs = recent['player_gsis_id'].unique()[:top_n].tolist()

        # Compute features for each and aggregate
        all_features = []
        for rb_id in top_rbs:
            rb_features = self.compute_rb_features(rb_id, season, week)
            if rb_features:
                all_features.append(rb_features)

        if not all_features:
            return {}

        # Average across top RBs
        aggregated = {}
        all_keys = set()
        for f in all_features:
            all_keys.update(f.keys())

        for key in all_keys:
            values = [f.get(key) for f in all_features if key in f]
            if values:
                aggregated[f'team_{key}'] = np.mean(values)

        return aggregated

    def compute_team_wr_features(
        self,
        team: str,
        season: int,
        week: int,
        top_n: int = 3
    ) -> Dict[str, float]:
        """
        Compute aggregated WR/TE features for a team.

        Args:
            team: Team abbreviation
            season: Season
            week: Week
            top_n: Number of top receivers to include

        Returns:
            Dict of aggregated WR features
        """
        self.load_data()

        if self.receiving_df.empty:
            return {}

        # Find team's WRs/TEs from past data
        mask = (
            (self.receiving_df['team_abbr'] == team) &
            (
                (self.receiving_df['season'] < season) |
                (
                    (self.receiving_df['season'] == season) &
                    (self.receiving_df['week'] < week) &
                    (self.receiving_df['week'] > 0)
                )
            )
        )
        team_wrs = self.receiving_df[mask]

        if team_wrs.empty:
            return {}

        # Get top WRs by targets
        recent = team_wrs.sort_values(['season', 'week']).groupby('player_gsis_id').tail(4)
        targets_col = 'targets' if 'targets' in recent.columns else None

        if targets_col:
            top_wrs = (
                recent.groupby('player_gsis_id')[targets_col]
                .sum()
                .nlargest(top_n)
                .index.tolist()
            )
        else:
            top_wrs = recent['player_gsis_id'].unique()[:top_n].tolist()

        # Compute features for each and aggregate
        all_features = []
        for wr_id in top_wrs:
            wr_features = self.compute_wr_features(wr_id, season, week)
            if wr_features:
                all_features.append(wr_features)

        if not all_features:
            return {}

        # Average across top WRs
        aggregated = {}
        all_keys = set()
        for f in all_features:
            all_keys.update(f.keys())

        for key in all_keys:
            values = [f.get(key) for f in all_features if key in f]
            if values:
                aggregated[f'team_{key}'] = np.mean(values)

        return aggregated


class NGSTeamAggregator:
    """
    Aggregate NGS features to team level for game prediction.

    For prediction, we need team-level features, not player-level.
    This aggregates across the team's skill position players.
    """

    def __init__(self, ngs_extractor: Optional[NGSFeatureExtractor] = None):
        """
        Initialize the team aggregator.

        Args:
            ngs_extractor: NGS feature extractor instance
        """
        self.extractor = ngs_extractor or NGSFeatureExtractor()

    def compute_team_offensive_features(
        self,
        team: str,
        season: int,
        week: int
    ) -> Dict[str, float]:
        """
        Compute all offensive NGS features for a team.

        Args:
            team: Team abbreviation
            season: Season
            week: Week

        Returns:
            Dict of team offensive features
        """
        features = {}

        # QB features (most important)
        qb_features = self.extractor.compute_team_qb_features(team, season, week)
        for k, v in qb_features.items():
            features[f'off_{k}'] = v

        # RB features
        rb_features = self.extractor.compute_team_rb_features(team, season, week)
        for k, v in rb_features.items():
            features[f'off_{k}'] = v

        # WR features
        wr_features = self.extractor.compute_team_wr_features(team, season, week)
        for k, v in wr_features.items():
            features[f'off_{k}'] = v

        return features

    def compute_matchup_features(
        self,
        home_team: str,
        away_team: str,
        season: int,
        week: int
    ) -> Dict[str, float]:
        """
        Compute NGS features for a specific matchup.

        Returns features for both teams plus differential features.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: Season
            week: Week

        Returns:
            Dict with home, away, and differential features
        """
        # Home team features
        home_features = self.compute_team_offensive_features(home_team, season, week)

        # Away team features
        away_features = self.compute_team_offensive_features(away_team, season, week)

        features = {}

        # Home features with prefix
        for k, v in home_features.items():
            features[f'home_{k}'] = v

        # Away features with prefix
        for k, v in away_features.items():
            features[f'away_{k}'] = v

        # Differential features (home - away)
        for key in home_features.keys():
            home_key = f'home_{key}'
            away_key = f'away_{key}'
            if home_key in features and away_key in features:
                h_val = features[home_key]
                a_val = features[away_key]
                if h_val is not None and a_val is not None:
                    features[f'diff_{key}'] = h_val - a_val

        return features


def main():
    """Test the feature extractor."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract NGS features")
    parser.add_argument('--team', type=str, default='KC',
                        help="Team to extract features for")
    parser.add_argument('--season', type=int, default=2024,
                        help="Season")
    parser.add_argument('--week', type=int, default=10,
                        help="Week")

    args = parser.parse_args()

    extractor = NGSFeatureExtractor()

    print(f"\n=== NGS Features for {args.team} (Season {args.season}, Week {args.week}) ===\n")

    # Get team QB features
    qb_features = extractor.compute_team_qb_features(args.team, args.season, args.week)
    if qb_features:
        print("QB Features:")
        for k, v in sorted(qb_features.items()):
            if v is not None:
                print(f"  {k}: {v:.4f}")
    else:
        print("No QB features available (check if NGS data is ingested)")

    # Get team RB features
    rb_features = extractor.compute_team_rb_features(args.team, args.season, args.week)
    if rb_features:
        print("\nRB Features:")
        for k, v in sorted(rb_features.items()):
            if v is not None:
                print(f"  {k}: {v:.4f}")

    # Get team WR features
    wr_features = extractor.compute_team_wr_features(args.team, args.season, args.week)
    if wr_features:
        print("\nWR Features:")
        for k, v in sorted(wr_features.items()):
            if v is not None:
                print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
