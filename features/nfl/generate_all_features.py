"""
Comprehensive Feature Generation Pipeline

Integrates all features from Phase 1-3 + QB Deviation Features:
- Baseline: Team form features (win rate, point differential, etc.)
- Phase 1: Schedule/rest features (days rest, bye weeks, travel, etc.)
- Phase 2: Injury features (QB status, O-line health, weighted impact, etc.)
- Phase 3: Weather features (temperature, wind, precipitation, conditions scores)
- QB Deviation: Career baselines, z-scores, trends, regression indicators

This is the main feature generation script for the full model.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.nfl.team_form_features import generate_team_form_features
from features.nfl.schedule_features import add_schedule_features_to_games
from features.nfl.injury_features import add_injury_features_to_games
from features.nfl.weather_features import add_weather_features_to_games
from ingestion.nfl.injuries_phase2 import InjuryIngestion
from ingestion.nfl.weather import WeatherIngestion
from ingestion.nfl.weather_parallel import ParallelWeatherIngestion, fetch_weather_parallel_batch

# Import QB deviation features (optional - may require play-by-play data)
try:
    from features.nfl.qb_deviation_features import generate_qb_deviation_features
    QB_DEVIATION_AVAILABLE = True
except ImportError as e:
    QB_DEVIATION_AVAILABLE = False
    _qb_import_error = str(e)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_base_data() -> tuple:
    """Load base games and team stats data."""
    project_root = Path(__file__).parent.parent.parent
    
    games_path = project_root / "data" / "nfl" / "staged" / "games.parquet"
    team_stats_path = project_root / "data" / "nfl" / "staged" / "team_stats.parquet"
    
    logger.info(f"Loading games from {games_path}")
    games_df = pd.read_parquet(games_path)
    
    logger.info(f"Loading team stats from {team_stats_path}")
    team_stats_df = pd.read_parquet(team_stats_path)
    
    return games_df, team_stats_df


def generate_baseline_features(games_df: pd.DataFrame) -> pd.DataFrame:
    """Generate baseline team form features."""
    logger.info("=" * 60)
    logger.info("Generating Baseline Features (Team Form)")
    logger.info("=" * 60)
    
    # Check if baseline features already exist
    project_root = Path(__file__).parent.parent.parent
    baseline_path = project_root / "data" / "nfl" / "processed" / "game_features_baseline.parquet"
    
    if baseline_path.exists():
        logger.info(f"Loading existing baseline features from {baseline_path}")
        baseline_df = pd.read_parquet(baseline_path)
        
        # Merge with games_df
        games_with_features = games_df.merge(
            baseline_df,
            on='game_id',
            how='left',
            suffixes=('', '_baseline')
        )
        
        # Remove duplicate columns
        for col in games_with_features.columns:
            if col.endswith('_baseline') and col.replace('_baseline', '') in games_with_features.columns:
                games_with_features = games_with_features.drop(columns=[col])
    else:
        logger.info("Baseline features not found, generating...")
        team_form_df = generate_team_form_features()
        
        # Merge team form features with games
        # Team form features are per-team, need to merge home and away separately
        home_features = team_form_df[team_form_df['is_home'] == True].copy()
        away_features = team_form_df[team_form_df['is_home'] == False].copy()
        
        # Rename columns for home team
        home_cols = {c: f'home_{c}' for c in home_features.columns if c not in ['game_id', 'team', 'is_home']}
        home_features = home_features.rename(columns=home_cols)
        
        # Rename columns for away team
        away_cols = {c: f'away_{c}' for c in away_features.columns if c not in ['game_id', 'team', 'is_home']}
        away_features = away_features.rename(columns=away_cols)
        
        # Merge
        games_with_features = games_df.merge(
            home_features[['game_id'] + [c for c in home_features.columns if c.startswith('home_')]],
            on='game_id',
            how='left'
        )
        
        games_with_features = games_with_features.merge(
            away_features[['game_id'] + [c for c in away_features.columns if c.startswith('away_')]],
            on='game_id',
            how='left'
        )
    
    baseline_feature_count = len([c for c in games_with_features.columns if 'last' in c])
    logger.info(f"Baseline features added: {baseline_feature_count} features")
    
    return games_with_features


def generate_schedule_features(games_df: pd.DataFrame) -> pd.DataFrame:
    """Generate Phase 1 schedule/rest features."""
    logger.info("=" * 60)
    logger.info("Generating Phase 1: Schedule/Rest Features")
    logger.info("=" * 60)
    
    try:
        games_with_schedule = add_schedule_features_to_games(games_df)
        schedule_features = [c for c in games_with_schedule.columns if any(x in c for x in ['rest', 'bye', 'travel', 'divisional', 'primetime', 'week_of'])]
        logger.info(f"Schedule features added: {len(schedule_features)} features")
        return games_with_schedule
    except Exception as e:
        logger.warning(f"Error adding schedule features: {e}")
        logger.warning("Continuing without schedule features...")
        return games_df


def generate_injury_features(games_df: pd.DataFrame, use_mock: bool = True) -> pd.DataFrame:
    """Generate Phase 2 injury features."""
    logger.info("=" * 60)
    logger.info("Generating Phase 2: Injury Features")
    logger.info("=" * 60)
    
    try:
        # Try to load or generate injury data
        injury_ingester = InjuryIngestion(source='auto')
        
        # Get seasons from games
        seasons = sorted(games_df['season'].unique())
        
        # Fetch historical injuries (will use mock if real data unavailable)
        injuries_df = injury_ingester.fetch_historical_injuries(
            seasons,
            games_df=games_df,
            use_mock_if_unavailable=use_mock,
        )
        
        if len(injuries_df) > 0:
            games_with_injuries = add_injury_features_to_games(games_df, injuries_df)
            injury_features = [c for c in games_with_injuries.columns if 'injury' in c or 'qb_status' in c or 'oline' in c]
            logger.info(f"Injury features added: {len(injury_features)} features")
            return games_with_injuries
        else:
            logger.warning("No injury data available, skipping injury features")
            return games_df
    except Exception as e:
        logger.warning(f"Error adding injury features: {e}")
        logger.warning("Continuing without injury features...")
        return games_df


def generate_weather_features(games_df: pd.DataFrame, use_cache: bool = True, use_parallel: bool = True) -> pd.DataFrame:
    """Generate Phase 3 weather features."""
    logger.info("=" * 60)
    logger.info("Generating Phase 3: Weather Features")
    logger.info("=" * 60)

    try:
        if use_parallel:
            logger.info("Using parallel weather fetching for faster processing...")
            # Fetch weather in parallel first
            weather_df = fetch_weather_parallel_batch(
                games_df,
                max_workers=10,  # Parallel API calls
            )

            # Then add features using the fetched weather data
            if len(weather_df) > 0:
                games_with_weather = add_weather_features_to_games(
                    games_df,
                    weather_df=weather_df,  # Use pre-fetched weather
                    weather_ingester=None,
                )
            else:
                logger.warning("No weather data fetched, using on-demand fetching")
                weather_ingester = WeatherIngestion()
                games_with_weather = add_weather_features_to_games(
                    games_df,
                    weather_df=None,
                    weather_ingester=weather_ingester,
                )
        else:
            # Sequential fetching (slower but more reliable)
            weather_ingester = WeatherIngestion()
            games_with_weather = add_weather_features_to_games(
                games_df,
                weather_df=None,
                weather_ingester=weather_ingester,
            )

        weather_features = [c for c in games_with_weather.columns if any(x in c for x in ['weather', 'temperature', 'wind', 'precipitation', 'dome', 'cold', 'hot', 'windy'])]
        logger.info(f"Weather features added: {len(weather_features)} features")
        return games_with_weather
    except Exception as e:
        logger.warning(f"Error adding weather features: {e}")
        logger.warning("Continuing without weather features...")
        return games_df


def add_qb_deviation_features(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add QB deviation-from-baseline features.

    These features capture:
    - Career baselines for each QB
    - Season-to-date performance
    - Deviation from career (z-scores)
    - Performance trends
    - Regression indicators

    Expected impact: +1-2% accuracy improvement
    """
    logger.info("=" * 60)
    logger.info("Generating QB Deviation Features")
    logger.info("=" * 60)

    if not QB_DEVIATION_AVAILABLE:
        logger.warning(f"QB deviation features not available: {_qb_import_error}")
        logger.warning("Continuing without QB deviation features...")
        return games_df

    try:
        # Generate QB deviation features
        project_root = Path(__file__).parent.parent.parent
        qb_features_path = project_root / "data" / "nfl" / "processed" / "qb_deviation_features.parquet"

        # Check if already generated
        if qb_features_path.exists():
            logger.info(f"Loading existing QB deviation features from {qb_features_path}")
            qb_df = pd.read_parquet(qb_features_path)
        else:
            logger.info("Generating QB deviation features (this may take a while)...")
            qb_df = generate_qb_deviation_features(games_df=games_df)

        # Merge home and away QB features
        home_qb = qb_df[qb_df['is_home'] == True].copy()
        away_qb = qb_df[qb_df['is_home'] == False].copy()

        # Rename columns for home team (qb_* -> home_qb_*)
        home_rename = {c: f'home_{c}' for c in home_qb.columns
                       if c.startswith('qb_') and c != 'qb_id'}
        home_rename['qb_id'] = 'home_qb_id'
        home_qb = home_qb.rename(columns=home_rename)

        # Rename columns for away team (qb_* -> away_qb_*)
        away_rename = {c: f'away_{c}' for c in away_qb.columns
                       if c.startswith('qb_') and c != 'qb_id'}
        away_rename['qb_id'] = 'away_qb_id'
        away_qb = away_qb.rename(columns=away_rename)

        # Merge with games
        home_cols = ['game_id'] + [c for c in home_qb.columns if c.startswith('home_')]
        away_cols = ['game_id'] + [c for c in away_qb.columns if c.startswith('away_')]

        games_with_qb = games_df.merge(
            home_qb[home_cols],
            on='game_id',
            how='left'
        )

        games_with_qb = games_with_qb.merge(
            away_qb[away_cols],
            on='game_id',
            how='left'
        )

        # Add differential features (home - away)
        qb_numeric_features = [c.replace('home_qb_', '') for c in home_qb.columns
                               if c.startswith('home_qb_') and c != 'home_qb_id']

        for feat in qb_numeric_features:
            home_col = f'home_qb_{feat}'
            away_col = f'away_qb_{feat}'
            if home_col in games_with_qb.columns and away_col in games_with_qb.columns:
                # Check if numeric
                if games_with_qb[home_col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    games_with_qb[f'qb_{feat}_diff'] = (
                        games_with_qb[home_col].fillna(0) -
                        games_with_qb[away_col].fillna(0)
                    )

        qb_features = [c for c in games_with_qb.columns if 'qb_' in c and c not in games_df.columns]
        logger.info(f"QB deviation features added: {len(qb_features)} features")

        return games_with_qb

    except Exception as e:
        logger.warning(f"Error adding QB deviation features: {e}")
        logger.warning("Continuing without QB deviation features...")
        return games_df


def generate_all_features(
    output_path: Optional[Path] = None,
    use_mock_injuries: bool = True,
    use_weather_cache: bool = True,
    include_qb_deviation: bool = True,
) -> pd.DataFrame:
    """
    Generate comprehensive feature set with all Phase 1-3 + QB Deviation features.

    Args:
        output_path: Path to save features (default: data/nfl/processed/game_features_phase3.parquet)
        use_mock_injuries: Use mock injury data if real data unavailable
        use_weather_cache: Use cached weather data when available
        include_qb_deviation: Include QB deviation-from-baseline features

    Returns:
        DataFrame with all features
    """
    logger.info("=" * 60)
    logger.info("COMPREHENSIVE FEATURE GENERATION")
    logger.info("Integrating Phase 1-3 + QB Deviation Features")
    logger.info("=" * 60)

    # Load base data
    games_df, team_stats_df = load_base_data()
    logger.info(f"Loaded {len(games_df)} games")

    # Step 1: Baseline features
    features_df = generate_baseline_features(games_df)

    # Step 2: Schedule features (Phase 1)
    features_df = generate_schedule_features(features_df)

    # Step 3: Injury features (Phase 2)
    features_df = generate_injury_features(features_df, use_mock=use_mock_injuries)

    # Step 4: Weather features (Phase 3) - using parallel fetching
    features_df = generate_weather_features(features_df, use_cache=use_weather_cache, use_parallel=True)

    # Step 5: QB Deviation features (if enabled)
    if include_qb_deviation:
        features_df = add_qb_deviation_features(features_df)

    # Ensure we have target variable
    if 'home_score' in features_df.columns and 'away_score' in features_df.columns:
        features_df['home_win'] = (features_df['home_score'] > features_df['away_score']).astype(int)

    # Save features
    if output_path is None:
        project_root = Path(__file__).parent.parent.parent
        output_path = project_root / "data" / "nfl" / "processed" / "game_features_phase3.parquet"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(output_path, index=False)

    logger.info("=" * 60)
    logger.info("FEATURE GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total features: {len(features_df.columns)}")
    logger.info(f"Total games: {len(features_df)}")
    logger.info(f"Saved to: {output_path}")

    # Count features by type
    baseline_count = len([c for c in features_df.columns if 'last' in c])
    schedule_count = len([c for c in features_df.columns if any(x in c for x in ['rest', 'bye', 'travel', 'divisional', 'primetime'])])
    injury_count = len([c for c in features_df.columns if 'injury' in c or 'qb_status' in c or 'oline' in c])
    weather_count = len([c for c in features_df.columns if any(x in c for x in ['weather', 'temperature', 'wind', 'precipitation', 'dome'])])
    qb_deviation_count = len([c for c in features_df.columns if 'qb_' in c and ('career' in c or 'season_' in c or 'zscore' in c or 'vs_career' in c or 'trend' in c or 'regression' in c)])

    logger.info("")
    logger.info("Feature breakdown:")
    logger.info(f"  Baseline (team form): {baseline_count}")
    logger.info(f"  Schedule/rest: {schedule_count}")
    logger.info(f"  Injury: {injury_count}")
    logger.info(f"  Weather: {weather_count}")
    logger.info(f"  QB Deviation: {qb_deviation_count}")

    return features_df


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate comprehensive features (Phase 1-3 + QB Deviation)")
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--no-mock-injuries', action='store_true', help='Do not use mock injury data')
    parser.add_argument('--no-weather-cache', action='store_true', help='Do not use cached weather data')
    parser.add_argument('--no-qb-deviation', action='store_true', help='Skip QB deviation features')
    parser.add_argument('--qb-deviation-only', action='store_true', help='Only generate QB deviation features')

    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None

    if args.qb_deviation_only:
        # Only generate QB deviation features
        logger.info("Generating QB deviation features only...")
        from features.nfl.qb_deviation_features import generate_qb_deviation_features
        generate_qb_deviation_features()
    else:
        generate_all_features(
            output_path=output_path,
            use_mock_injuries=not args.no_mock_injuries,
            use_weather_cache=not args.no_weather_cache,
            include_qb_deviation=not args.no_qb_deviation,
        )


if __name__ == "__main__":
    main()

