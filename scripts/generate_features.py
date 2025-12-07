"""
Generate features for specific seasons and weeks for prediction and simulation.

Loads schedule and team-level data (odds, rosters, injuries, stats) for specified games,
generates full feature table using production feature pipeline, and saves outputs.

Usage:
    python3 scripts/generate_features.py --seasons 2025 --weeks 1-18
    python3 scripts/generate_features.py --seasons 2023,2024 --weeks 14-18
    python3 scripts/generate_features.py --seasons 2025 --weeks 14
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Set
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.pipelines.feature_pipeline import (
    merge_team_features_to_games,
    run_baseline_feature_pipeline,
)
from features.nfl.team_form_features import generate_team_form_features
from ingestion.nfl.schedule import form_game_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_weeks(weeks_str: str) -> Set[int]:
    """
    Parse weeks string into set of week numbers.
    
    Supports:
    - Single week: "14"
    - Range: "1-18"
    - List: "1,2,3" or "14,15,16"
    - Combination: "1-18,20-22" (for playoffs)
    
    Args:
        weeks_str: Weeks string
    
    Returns:
        Set of week numbers
    """
    weeks = set()
    
    for part in weeks_str.split(','):
        part = part.strip()
        if '-' in part:
            # Range: "1-18"
            start, end = part.split('-')
            weeks.update(range(int(start), int(end) + 1))
        else:
            # Single week
            weeks.add(int(part))
    
    return weeks


def parse_seasons(seasons_str: str) -> List[int]:
    """
    Parse seasons string into list of season years.
    
    Supports:
    - Single season: "2025"
    - List: "2023,2024,2025"
    
    Args:
        seasons_str: Seasons string
    
    Returns:
        List of season years
    """
    return [int(s.strip()) for s in seasons_str.split(',')]


def check_data_availability(
    games_df: pd.DataFrame,
    markets_df: Optional[pd.DataFrame],
    team_stats_df: Optional[pd.DataFrame],
    seasons: List[int],
    weeks: Set[int],
) -> dict:
    """
    Check data availability for specified games.
    
    Args:
        games_df: Games dataframe
        markets_df: Markets dataframe (optional)
        team_stats_df: Team stats dataframe (optional)
        seasons: List of seasons
        weeks: Set of weeks
    
    Returns:
        Dictionary with availability status
    """
    # Filter games
    filtered_games = games_df[
        (games_df['season'].isin(seasons)) &
        (games_df['week'].isin(weeks))
    ].copy()
    
    status = {
        'total_games': len(filtered_games),
        'games_with_schedule': len(filtered_games),
        'games_with_odds': 0,
        'games_with_team_stats': 0,
        'missing_odds': [],
        'missing_team_stats': [],
    }
    
    if markets_df is not None:
        games_with_odds = filtered_games[
            filtered_games['game_id'].isin(markets_df['game_id'].unique())
        ]
        status['games_with_odds'] = len(games_with_odds)
        status['missing_odds'] = filtered_games[
            ~filtered_games['game_id'].isin(markets_df['game_id'].unique())
        ]['game_id'].tolist()
    
    if team_stats_df is not None:
        # Check if team stats exist for games (need both home and away)
        games_with_stats = []
        for _, game in filtered_games.iterrows():
            home_stats = team_stats_df[
                (team_stats_df['game_id'] == game['game_id']) &
                (team_stats_df['team'] == game['home_team'])
            ]
            away_stats = team_stats_df[
                (team_stats_df['game_id'] == game['game_id']) &
                (team_stats_df['team'] == game['away_team'])
            ]
            if len(home_stats) > 0 and len(away_stats) > 0:
                games_with_stats.append(game['game_id'])
        
        status['games_with_team_stats'] = len(games_with_stats)
        status['missing_team_stats'] = [
            gid for gid in filtered_games['game_id'].tolist()
            if gid not in games_with_stats
        ]
    
    return status


def generate_features_for_season_week(
    season: int,
    week: int,
    games_markets_df: pd.DataFrame,
    team_features_df: pd.DataFrame,
    output_dir: Path,
) -> Optional[pd.DataFrame]:
    """
    Generate features for a specific season and week.
    
    Args:
        season: Season year
        week: Week number
        games_markets_df: Full games_markets dataframe
        team_features_df: Full team features dataframe
        output_dir: Output directory
    
    Returns:
        DataFrame with features for the season/week, or None if no games
    """
    # Filter games for this season/week
    games_filtered = games_markets_df[
        (games_markets_df['season'] == season) &
        (games_markets_df['week'] == week)
    ].copy()
    
    if len(games_filtered) == 0:
        logger.warning(f"No games found for {season} Week {week}")
        return None
    
    logger.info(f"Processing {season} Week {week}: {len(games_filtered)} games")
    
    # Filter team features for these games
    team_features_filtered = team_features_df[
        team_features_df['game_id'].isin(games_filtered['game_id'].unique())
    ].copy()
    
    if len(team_features_filtered) == 0:
        logger.warning(f"No team features found for {season} Week {week}")
        return None
    
    # Separate home and away team features
    home_features = team_features_filtered[team_features_filtered["is_home"] == True].copy()
    away_features = team_features_filtered[team_features_filtered["is_home"] == False].copy()
    
    # Prefix feature columns
    feature_cols = [col for col in team_features_df.columns 
                   if col not in ["game_id", "team", "is_home"]]
    
    # Rename home team features
    home_feature_map = {col: f"home_{col}" for col in feature_cols}
    home_features = home_features.rename(columns=home_feature_map)
    
    # Rename away team features
    away_feature_map = {col: f"away_{col}" for col in feature_cols}
    away_features = away_features.rename(columns=away_feature_map)
    
    # Merge home team features
    result_df = games_filtered.merge(
        home_features[["game_id"] + [f"home_{col}" for col in feature_cols]],
        on="game_id",
        how="left",
    )
    
    # Merge away team features
    result_df = result_df.merge(
        away_features[["game_id"] + [f"away_{col}" for col in feature_cols]],
        on="game_id",
        how="left",
    )
    
    # Check for missing features
    missing_home = result_df[[f"home_{col}" for col in feature_cols]].isna().any(axis=1).sum()
    missing_away = result_df[[f"away_{col}" for col in feature_cols]].isna().any(axis=1).sum()
    
    if missing_home > 0:
        logger.warning(f"  {missing_home} games missing home team features")
    if missing_away > 0:
        logger.warning(f"  {missing_away} games missing away team features")
    
    # Fill missing values with 0 (for games without historical data)
    result_df = result_df.fillna(0)
    
    # Sort by date
    result_df = result_df.sort_values(["date"]).reset_index(drop=True)
    
    # Save to file
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"features_{season}_wk{week:02d}.parquet"
    result_df.to_parquet(output_path, index=False)
    logger.info(f"  Saved {len(result_df)} games to {output_path}")
    
    return result_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate features for specific seasons and weeks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate features for 2025 season, weeks 1-18
  python3 scripts/generate_features.py --seasons 2025 --weeks 1-18
  
  # Generate features for multiple seasons, specific weeks
  python3 scripts/generate_features.py --seasons 2023,2024 --weeks 14-18
  
  # Generate features for single week
  python3 scripts/generate_features.py --seasons 2025 --weeks 14
  
  # Generate features for playoffs
  python3 scripts/generate_features.py --seasons 2024 --weeks 19-22
        """
    )
    parser.add_argument(
        '--seasons',
        type=str,
        required=True,
        help='Comma-separated list of seasons (e.g., "2025" or "2023,2024,2025")'
    )
    parser.add_argument(
        '--weeks',
        type=str,
        required=True,
        help='Weeks to process (e.g., "1-18", "14", "14,15,16", or "1-18,20-22")'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: data/nfl/features/)'
    )
    parser.add_argument(
        '--feature-table',
        type=str,
        default='baseline',
        help='Feature table type: baseline, phase2, phase2b (default: baseline)'
    )
    
    args = parser.parse_args()
    
    # Parse arguments
    seasons = parse_seasons(args.seasons)
    weeks = parse_weeks(args.weeks)
    
    logger.info("=" * 80)
    logger.info("Feature Generation for Prediction and Simulation")
    logger.info("=" * 80)
    logger.info(f"Seasons: {seasons}")
    logger.info(f"Weeks: {sorted(weeks)}")
    logger.info(f"Feature Table: {args.feature_table}")
    
    # Set up paths
    project_root = Path(__file__).parent.parent
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "data" / "nfl" / "features"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output Directory: {output_dir}")
    
    # Load data sources
    logger.info("\n[Step 1/4] Loading data sources...")
    
    games_path = project_root / "data" / "nfl" / "staged" / "games.parquet"
    markets_path = project_root / "data" / "nfl" / "staged" / "markets.parquet"
    games_markets_path = project_root / "data" / "nfl" / "staged" / "games_markets.parquet"
    team_stats_path = project_root / "data" / "nfl" / "staged" / "team_stats.parquet"
    
    # Load games
    if not games_path.exists():
        raise FileNotFoundError(f"Games file not found: {games_path}")
    games_df = pd.read_parquet(games_path)
    logger.info(f"  Loaded {len(games_df)} games from schedule")
    
    # Load games_markets (or create from games + markets)
    if games_markets_path.exists():
        games_markets_df = pd.read_parquet(games_markets_path)
        logger.info(f"  Loaded {len(games_markets_df)} games with markets")
    else:
        logger.info("  games_markets.parquet not found, creating from games + markets...")
        games_markets_df = games_df.copy()
        
        if markets_path.exists():
            markets_df = pd.read_parquet(markets_path)
            games_markets_df = games_markets_df.merge(
                markets_df[['game_id', 'close_spread', 'close_total', 'open_spread', 'open_total']],
                on='game_id',
                how='left'
            )
            logger.info(f"  Merged markets data: {games_markets_df['close_spread'].notna().sum()} games with odds")
        else:
            logger.warning(f"  Markets file not found: {markets_path}")
            # Add empty columns
            games_markets_df['close_spread'] = np.nan
            games_markets_df['close_total'] = np.nan
            games_markets_df['open_spread'] = np.nan
            games_markets_df['open_total'] = np.nan
    
    # Load team stats (optional)
    team_stats_df = None
    if team_stats_path.exists():
        team_stats_df = pd.read_parquet(team_stats_path)
        logger.info(f"  Loaded {len(team_stats_df)} team-game stats")
    else:
        logger.warning(f"  Team stats file not found: {team_stats_path}")
    
    # Check data availability
    logger.info("\n[Step 2/4] Checking data availability...")
    markets_df = None
    if markets_path.exists():
        markets_df = pd.read_parquet(markets_path)
    
    availability = check_data_availability(
        games_df, markets_df, team_stats_df, seasons, weeks
    )
    
    logger.info(f"  Total games: {availability['total_games']}")
    logger.info(f"  Games with schedule: {availability['games_with_schedule']}")
    logger.info(f"  Games with odds: {availability['games_with_odds']}")
    logger.info(f"  Games with team stats: {availability['games_with_team_stats']}")
    
    if availability['missing_odds']:
        logger.warning(f"  Missing odds for {len(availability['missing_odds'])} games")
        if len(availability['missing_odds']) <= 10:
            logger.warning(f"    Examples: {availability['missing_odds'][:5]}")
    
    if availability['missing_team_stats']:
        logger.warning(f"  Missing team stats for {len(availability['missing_team_stats'])} games")
        if len(availability['missing_team_stats']) <= 10:
            logger.warning(f"    Examples: {availability['missing_team_stats'][:5]}")
    
    # Generate team features (if needed)
    logger.info("\n[Step 3/4] Generating team form features...")
    team_features_df = generate_team_form_features()
    logger.info(f"  Generated team features for {len(team_features_df)} team-games")
    
    # Generate features for each season/week combination
    logger.info("\n[Step 4/4] Generating game-level features...")
    
    results_summary = []
    
    for season in seasons:
        for week in sorted(weeks):
            try:
                features_df = generate_features_for_season_week(
                    season, week, games_markets_df, team_features_df, output_dir
                )
                
                if features_df is not None:
                    results_summary.append({
                        'season': season,
                        'week': week,
                        'games': len(features_df),
                        'status': 'success'
                    })
                else:
                    results_summary.append({
                        'season': season,
                        'week': week,
                        'games': 0,
                        'status': 'no_games'
                    })
            except Exception as e:
                logger.error(f"Error generating features for {season} Week {week}: {e}")
                results_summary.append({
                    'season': season,
                    'week': week,
                    'games': 0,
                    'status': f'error: {str(e)[:50]}'
                })
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Feature Generation Summary")
    logger.info("=" * 80)
    
    successful = [r for r in results_summary if r['status'] == 'success']
    failed = [r for r in results_summary if r['status'] != 'success']
    
    total_games = sum(r['games'] for r in successful)
    
    logger.info(f"\nSuccessfully generated features:")
    logger.info(f"  Files created: {len(successful)}")
    logger.info(f"  Total games: {total_games}")
    
    if successful:
        logger.info(f"\n  Per file breakdown:")
        for r in successful:
            logger.info(f"    {r['season']} Week {r['week']:2d}: {r['games']} games -> features_{r['season']}_wk{r['week']:02d}.parquet")
    
    if failed:
        logger.warning(f"\nFailed or skipped:")
        for r in failed:
            logger.warning(f"    {r['season']} Week {r['week']:2d}: {r['status']}")
    
    logger.info(f"\nOutput directory: {output_dir}")
    logger.info("=" * 80)
    
    return results_summary


if __name__ == "__main__":
    main()

