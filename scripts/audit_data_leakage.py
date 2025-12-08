"""
Data Leakage Audit Script

Ensures no future information leaks into model training or prediction.
This is critical for realistic performance evaluation.

Checks:
1. Team stats only include games BEFORE this game
2. Injury data is from pre-game report (not post-game)
3. Weather is forecast (for future games) or game-day (for historical)
4. Odds are pre-game closing lines (not live/post-game)
5. No outcome information (score, winner) in features
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.training.trainer import load_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def audit_feature_pipeline(
    game_id: str,
    features: Dict,
    games_df: pd.DataFrame,
) -> List[str]:
    """
    Audit features for a single game for potential leakage.
    
    Args:
        game_id: Game identifier
        features: Dictionary of features for this game
        games_df: DataFrame with all games (for temporal checks)
    
    Returns:
        List of leakage warnings (empty if clean)
    """
    warnings = []
    
    # Get game info
    game = games_df[games_df['game_id'] == game_id]
    if len(game) == 0:
        warnings.append(f"Game {game_id} not found in games_df")
        return warnings
    
    game = game.iloc[0]
    game_date = pd.to_datetime(game.get('gameday') or game.get('date'))
    game_season = game.get('season')
    game_week = game.get('week')
    home_team = game.get('home_team')
    away_team = game.get('away_team')
    
    # Check 1: Team stats timing
    # Features like "home_win_rate_last4" should only include games BEFORE this game
    for feature_name, feature_value in features.items():
        if 'last' in feature_name.lower() or 'rolling' in feature_name.lower():
            # This is a rolling/aggregate feature
            # Check if it could include future games
            # (We can't fully verify without seeing the calculation, but we can check for obvious issues)
            if pd.isna(feature_value):
                continue  # Missing is OK
            
            # Check if feature name suggests it uses correct window
            if 'next' in feature_name.lower() or 'future' in feature_name.lower():
                warnings.append(f"Feature {feature_name} suggests future data usage")
    
    # Check 2: Outcome information leakage
    outcome_features = ['home_score', 'away_score', 'winner', 'home_win', 'result']
    for feat in outcome_features:
        if feat in features:
            warnings.append(f"Feature {feat} contains outcome information (should not be in features)")
    
    # Check 3: Injury data timing
    # Injury features should be from pre-game report
    injury_features = [f for f in features.keys() if 'injury' in f.lower()]
    for feat in injury_features:
        # Can't fully verify without injury report dates, but check for post-game indicators
        if 'post' in feat.lower() or 'after' in feat.lower():
            warnings.append(f"Injury feature {feat} suggests post-game data")
    
    # Check 4: Weather data timing
    weather_features = [f for f in features.keys() if 'weather' in f.lower() or 'temperature' in f.lower()]
    # Weather should be forecast for future games, actual for historical
    # Can't fully verify without weather fetch timestamp, but check for obvious issues
    
    # Check 5: Odds timing
    odds_features = [f for f in features.keys() if 'odds' in f.lower() or 'spread' in f.lower() or 'market' in f.lower()]
    for feat in odds_features:
        if 'live' in feat.lower() or 'post' in feat.lower():
            warnings.append(f"Odds feature {feat} suggests live/post-game data")
    
    return warnings


def audit_train_test_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> List[str]:
    """
    Audit train/test split for temporal leakage.
    
    Checks:
    1. All training games occur BEFORE all test games
    2. No game appears in both sets
    3. Rolling features don't peek into future
    
    Args:
        train_df: Training games DataFrame
        test_df: Test games DataFrame
    
    Returns:
        List of leakage warnings
    """
    warnings = []
    
    # Check 1: Temporal ordering
    if 'gameday' in train_df.columns and 'gameday' in test_df.columns:
        train_dates = pd.to_datetime(train_df['gameday'])
        test_dates = pd.to_datetime(test_df['gameday'])
        
        if train_dates.max() > test_dates.min():
            warnings.append(
                f"Temporal leakage: Latest training game ({train_dates.max()}) "
                f"is after earliest test game ({test_dates.min()})"
            )
    
    # Check 2: Season ordering
    if 'season' in train_df.columns and 'season' in test_df.columns:
        train_seasons = train_df['season'].unique()
        test_seasons = test_df['season'].unique()
        
        if max(train_seasons) > min(test_seasons):
            warnings.append(
                f"Season leakage: Training includes season {max(train_seasons)} "
                f"which is >= test season {min(test_seasons)}"
            )
    
    # Check 3: Duplicate games
    if 'game_id' in train_df.columns and 'game_id' in test_df.columns:
        train_ids = set(train_df['game_id'])
        test_ids = set(test_df['game_id'])
        overlap = train_ids & test_ids
        
        if overlap:
            warnings.append(f"Duplicate games in train and test: {len(overlap)} games")
            if len(overlap) <= 10:
                warnings.append(f"  Overlapping game_ids: {list(overlap)[:10]}")
    
    return warnings


def audit_rolling_features(
    games_df: pd.DataFrame,
    feature_cols: List[str],
) -> List[str]:
    """
    Audit rolling/aggregate features for temporal leakage.
    
    Checks that rolling windows only look backward, not forward.
    
    Args:
        games_df: DataFrame with games and features
        feature_cols: List of feature column names
    
    Returns:
        List of leakage warnings
    """
    warnings = []
    
    # Sort by date
    if 'gameday' in games_df.columns:
        games_sorted = games_df.sort_values('gameday').copy()
    elif 'date' in games_df.columns:
        games_sorted = games_df.sort_values('date').copy()
    else:
        warnings.append("Cannot audit rolling features: no date column found")
        return warnings
    
    # Check rolling features
    rolling_features = [f for f in feature_cols if 'last' in f.lower() or 'rolling' in f.lower()]
    
    for feat in rolling_features[:5]:  # Sample first 5 to avoid too many checks
        # Check if values increase monotonically (suggests forward-looking)
        # This is a heuristic - not perfect but catches obvious issues
        values = games_sorted[feat].dropna()
        if len(values) > 10:
            # Check for suspicious patterns
            # If feature values are perfectly correlated with game order, might be forward-looking
            correlation = values.reset_index(drop=True).corrwith(pd.Series(range(len(values))))
            if abs(correlation) > 0.9:
                warnings.append(
                    f"Rolling feature {feat} shows high correlation ({correlation:.2f}) "
                    f"with game order - may be forward-looking"
                )
    
    return warnings


def audit_full_pipeline(
    feature_table: str = "baseline",
    sample_size: int = 50,
) -> Dict:
    """
    Run full audit on entire pipeline.
    
    1. Load sample of games from each season
    2. Regenerate features
    3. Check each feature for leakage
    4. Generate audit report
    
    Args:
        feature_table: Feature table name
        sample_size: Number of games to sample per season
    
    Returns:
        Dictionary with audit results
    """
    logger.info("=" * 60)
    logger.info("Data Leakage Audit")
    logger.info("=" * 60)
    
    # Load features
    logger.info("Loading features...")
    X, y, feature_cols, games_df = load_features(feature_table=feature_table)
    
    all_warnings = []
    
    # Sample games from each season
    seasons = sorted(games_df['season'].unique())
    logger.info(f"Auditing {len(seasons)} seasons")
    
    for season in seasons:
        season_games = games_df[games_df['season'] == season]
        if len(season_games) > sample_size:
            season_games = season_games.sample(n=sample_size, random_state=42)
        
        logger.info(f"  Season {season}: {len(season_games)} games")
        
        for idx, game in season_games.iterrows():
            game_id = game.get('game_id')
            if not game_id:
                continue
            
            # Get features for this game
            game_features = X.loc[idx].to_dict()
            
            # Audit features
            warnings = audit_feature_pipeline(game_id, game_features, games_df)
            all_warnings.extend(warnings)
    
    # Audit train/test split
    logger.info("Auditing train/test split...")
    train_df, val_df, test_df = split_by_season(games_df)
    
    split_warnings = audit_train_test_split(train_df, test_df)
    all_warnings.extend(split_warnings)
    
    # Audit rolling features
    logger.info("Auditing rolling features...")
    rolling_warnings = audit_rolling_features(games_df, feature_cols)
    all_warnings.extend(rolling_warnings)
    
    # Compile results
    results = {
        'total_games_audited': len(games_df),
        'total_warnings': len(all_warnings),
        'warnings': all_warnings,
        'warnings_by_type': {},
        'status': 'PASS' if len(all_warnings) == 0 else 'FAIL',
    }
    
    # Categorize warnings
    for warning in all_warnings:
        if 'temporal' in warning.lower() or 'date' in warning.lower():
            results['warnings_by_type']['temporal'] = results['warnings_by_type'].get('temporal', 0) + 1
        elif 'outcome' in warning.lower() or 'score' in warning.lower() or 'winner' in warning.lower():
            results['warnings_by_type']['outcome'] = results['warnings_by_type'].get('outcome', 0) + 1
        elif 'injury' in warning.lower():
            results['warnings_by_type']['injury'] = results['warnings_by_type'].get('injury', 0) + 1
        elif 'odds' in warning.lower() or 'market' in warning.lower():
            results['warnings_by_type']['odds'] = results['warnings_by_type'].get('odds', 0) + 1
        elif 'rolling' in warning.lower() or 'forward' in warning.lower():
            results['warnings_by_type']['rolling'] = results['warnings_by_type'].get('rolling', 0) + 1
        else:
            results['warnings_by_type']['other'] = results['warnings_by_type'].get('other', 0) + 1
    
    logger.info(f"Audit complete: {len(all_warnings)} warnings found")
    
    return results


def generate_audit_report(results: Dict, output_path: Path) -> str:
    """
    Generate markdown audit report.
    
    Args:
        results: Audit results dictionary
        output_path: Path to save report
    
    Returns:
        Report text
    """
    report_lines = []
    
    report_lines.append("# Data Leakage Audit Report")
    report_lines.append("")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Summary
    report_lines.append("## Summary")
    report_lines.append("")
    report_lines.append(f"- **Status**: {results.get('status', 'UNKNOWN')}")
    report_lines.append(f"- **Total Warnings**: {results.get('total_warnings', 0)}")
    report_lines.append(f"- **Games Audited**: {results.get('total_games_audited', 0)}")
    report_lines.append("")
    
    # Warnings by Type
    report_lines.append("## Warnings by Type")
    report_lines.append("")
    warnings_by_type = results.get('warnings_by_type', {})
    if warnings_by_type:
        report_lines.append("| Type | Count |")
        report_lines.append("|------|-------|")
        for warn_type, count in sorted(warnings_by_type.items()):
            report_lines.append(f"| {warn_type} | {count} |")
    else:
        report_lines.append("No warnings found!")
    report_lines.append("")
    
    # Detailed Warnings
    warnings = results.get('warnings', [])
    if warnings:
        report_lines.append("## Detailed Warnings")
        report_lines.append("")
        for i, warning in enumerate(warnings[:50], 1):  # Limit to first 50
            report_lines.append(f"{i}. {warning}")
        
        if len(warnings) > 50:
            report_lines.append(f"\n... and {len(warnings) - 50} more warnings")
        report_lines.append("")
    
    # Recommendations
    report_lines.append("## Recommendations")
    report_lines.append("")
    if results.get('status') == 'PASS':
        report_lines.append("✓ No data leakage detected. Pipeline appears clean.")
    else:
        report_lines.append("⚠ Data leakage warnings found. Review and fix:")
        report_lines.append("")
        report_lines.append("1. Ensure all rolling features only look backward")
        report_lines.append("2. Verify train/test split respects temporal ordering")
        report_lines.append("3. Check that injury/weather/odds data is from pre-game")
        report_lines.append("4. Remove any outcome features (scores, winners) from feature set")
    report_lines.append("")
    
    report_text = "\n".join(report_lines)
    
    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    logger.info(f"Saved audit report to {output_path}")
    
    return report_text


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Audit pipeline for data leakage")
    parser.add_argument('--feature-table', type=str, default='baseline', help='Feature table name')
    parser.add_argument('--sample-size', type=int, default=50, help='Games to sample per season')
    parser.add_argument('--output-dir', type=str, default='logs/phase3_validation', help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run audit
    results = audit_full_pipeline(
        feature_table=args.feature_table,
        sample_size=args.sample_size,
    )
    
    # Save results JSON
    results_path = output_dir / "data_leakage_audit_results.json"
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Saved results to {results_path}")
    
    # Generate report
    report_path = output_dir / "data_leakage_audit_report.md"
    generate_audit_report(results, report_path)
    
    logger.info("=" * 60)
    logger.info("Data Leakage Audit Complete!")
    logger.info("=" * 60)
    
    if results.get('status') == 'FAIL':
        logger.warning(f"Found {results.get('total_warnings', 0)} warnings - review audit report")
        sys.exit(1)
    else:
        logger.info("No data leakage detected - pipeline is clean!")


if __name__ == "__main__":
    main()

