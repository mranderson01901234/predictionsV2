"""
Sample Phase 1 Pipeline

Runs the complete Phase 1 baseline pipeline on sample data.
This is designed for CI testing and quick smoke tests without external dependencies.

Usage:
    python -m orchestration.pipelines.phase1_sample_pipeline
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import logging
from typing import Tuple, Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Sample data paths
SAMPLE_DIR = Path(__file__).parent.parent.parent / "data" / "nfl" / "sample"
STAGED_DIR = SAMPLE_DIR / "staged"
PROCESSED_DIR = SAMPLE_DIR / "processed"


def load_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load sample data from CSV files.

    Returns:
        Tuple of (games_df, markets_df, team_stats_df)
    """
    logger.info("Loading sample data from CSV files...")

    # Try CSV first, then parquet
    games_path = STAGED_DIR / "games.csv"
    if games_path.exists():
        games_df = pd.read_csv(games_path, parse_dates=["date"])
    else:
        games_df = pd.read_parquet(STAGED_DIR / "games.parquet")

    markets_path = STAGED_DIR / "markets.csv"
    if markets_path.exists():
        markets_df = pd.read_csv(markets_path)
    else:
        markets_df = pd.read_parquet(STAGED_DIR / "markets.parquet")

    team_stats_path = STAGED_DIR / "team_stats.csv"
    if team_stats_path.exists():
        team_stats_df = pd.read_csv(team_stats_path)
    else:
        team_stats_df = pd.read_parquet(STAGED_DIR / "team_stats.parquet")

    logger.info(f"  Loaded {len(games_df)} games")
    logger.info(f"  Loaded {len(markets_df)} market entries")
    logger.info(f"  Loaded {len(team_stats_df)} team stat entries")

    return games_df, markets_df, team_stats_df


def join_games_markets(games_df: pd.DataFrame, markets_df: pd.DataFrame) -> pd.DataFrame:
    """Join games and markets dataframes."""
    logger.info("Joining games and markets...")
    joined_df = games_df.merge(
        markets_df[["game_id", "close_spread", "close_total"]], on="game_id", how="left"
    )
    logger.info(f"  Joined {len(joined_df)} games with markets")
    return joined_df


def calculate_rolling_features(team_stats_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rolling team form features.

    Uses the same logic as features/nfl/team_form_features.py but simplified for sample data.
    """
    logger.info("Calculating rolling team form features...")

    # Merge with games to get date
    df = team_stats_df.merge(games_df[["game_id", "date"]], on="game_id", how="left")

    # Calculate win/loss
    df["win"] = (df["points_for"] > df["points_against"]).astype(float)
    df.loc[df["points_for"] == df["points_against"], "win"] = 0.5
    df["point_diff"] = df["points_for"] - df["points_against"]

    # Sort by team and date
    df = df.sort_values(["team", "date"]).reset_index(drop=True)

    result_rows = []
    windows = [4, 8, 16]

    for team in df["team"].unique():
        team_df = df[df["team"] == team].copy().reset_index(drop=True)

        for idx, row in team_df.iterrows():
            historical = team_df.iloc[:idx]  # Exclude current game
            feature_row = row.to_dict()

            for window in windows:
                window_data = historical.tail(window)

                if len(window_data) > 0:
                    feature_row[f"win_rate_last{window}"] = window_data["win"].mean()
                    feature_row[f"pdiff_last{window}"] = window_data["point_diff"].mean()
                    feature_row[f"points_for_last{window}"] = window_data["points_for"].mean()
                    feature_row[f"points_against_last{window}"] = window_data["points_against"].mean()
                else:
                    feature_row[f"win_rate_last{window}"] = 0.0
                    feature_row[f"pdiff_last{window}"] = 0.0
                    feature_row[f"points_for_last{window}"] = 0.0
                    feature_row[f"points_against_last{window}"] = 0.0

            result_rows.append(feature_row)

    features_df = pd.DataFrame(result_rows)
    logger.info(f"  Generated features for {len(features_df)} team-games")
    return features_df


def merge_features_to_games(
    games_markets_df: pd.DataFrame, team_features_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge team features into game-level feature table."""
    logger.info("Merging team features into game-level table...")

    # Feature columns to merge
    feature_cols = [
        col for col in team_features_df.columns
        if col not in ["game_id", "team", "is_home", "points_for", "points_against",
                       "turnovers", "yards_for", "yards_against", "win", "point_diff", "date"]
    ]

    # Split by home/away
    home_features = team_features_df[team_features_df["is_home"] == True].copy()
    away_features = team_features_df[team_features_df["is_home"] == False].copy()

    # Rename columns
    home_features = home_features.rename(columns={col: f"home_{col}" for col in feature_cols})
    away_features = away_features.rename(columns={col: f"away_{col}" for col in feature_cols})

    # Merge home features
    result_df = games_markets_df.merge(
        home_features[["game_id"] + [f"home_{col}" for col in feature_cols]],
        on="game_id",
        how="left",
    )

    # Merge away features
    result_df = result_df.merge(
        away_features[["game_id"] + [f"away_{col}" for col in feature_cols]],
        on="game_id",
        how="left",
    )

    logger.info(f"  Created game-level features: {len(result_df)} games, {len(result_df.columns)} columns")
    return result_df


def train_simple_model(game_features_df: pd.DataFrame) -> dict:
    """
    Train a simple logistic regression model and return metrics.

    Uses weeks 1-4 for training, weeks 5-6 for testing.
    """
    logger.info("Training simple baseline model...")

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

    # Create target
    game_features_df = game_features_df.copy()
    game_features_df["home_win"] = (
        game_features_df["home_score"] > game_features_df["away_score"]
    ).astype(int)

    # Feature columns
    feature_cols = [
        col for col in game_features_df.columns
        if col.startswith(("home_", "away_")) and "score" not in col
    ]

    # Split by week
    train_df = game_features_df[game_features_df["week"] <= 4].copy()
    test_df = game_features_df[game_features_df["week"] >= 5].copy()

    if len(train_df) < 4 or len(test_df) < 4:
        logger.warning("Not enough data for train/test split")
        return {"accuracy": 0.0, "brier_score": 1.0, "log_loss": 1.0, "n_train": len(train_df), "n_test": len(test_df)}

    # Fill NaN with 0
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["home_win"]
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df["home_win"]

    # Train model
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_proba)
    logloss = log_loss(y_test, y_proba)

    logger.info(f"  Training set: {len(X_train)} games")
    logger.info(f"  Test set: {len(X_test)} games")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Brier Score: {brier:.4f}")
    logger.info(f"  Log Loss: {logloss:.4f}")

    return {
        "accuracy": accuracy,
        "brier_score": brier,
        "log_loss": logloss,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def generate_sample_report(metrics: dict, output_path: Path) -> None:
    """Generate a sample report markdown file."""
    report = f"""# Sample Phase 1 Baseline Report

## Overview
This report summarizes the results of running the Phase 1 baseline pipeline on sample data.

## Data Summary
- Training games: {metrics['n_train']}
- Test games: {metrics['n_test']}

## Model Performance (Logistic Regression)

| Metric | Value |
|--------|-------|
| Accuracy | {metrics['accuracy']:.4f} |
| Brier Score | {metrics['brier_score']:.4f} |
| Log Loss | {metrics['log_loss']:.4f} |

## Notes
- This is a smoke test using synthetic sample data
- Results are not representative of real-world performance
- The purpose is to verify pipeline correctness
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    logger.info(f"  Report saved to {output_path}")


def run_sample_pipeline() -> dict:
    """
    Run the complete sample Phase 1 pipeline.

    Returns:
        Dictionary with pipeline results and metrics
    """
    logger.info("=" * 60)
    logger.info("Sample Phase 1 Pipeline")
    logger.info("=" * 60)

    # Ensure output directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load sample data
    logger.info("\n[Step 1/5] Loading sample data...")
    games_df, markets_df, team_stats_df = load_sample_data()

    # Step 2: Join games and markets
    logger.info("\n[Step 2/5] Joining games and markets...")
    games_markets_df = join_games_markets(games_df, markets_df)

    # Step 3: Calculate rolling features
    logger.info("\n[Step 3/5] Calculating rolling features...")
    team_features_df = calculate_rolling_features(team_stats_df, games_df)

    # Step 4: Merge features to games
    logger.info("\n[Step 4/5] Merging features to games...")
    game_features_df = merge_features_to_games(games_markets_df, team_features_df)

    # Save processed data
    game_features_df.to_parquet(PROCESSED_DIR / "game_features_sample.parquet", index=False)
    team_features_df.to_parquet(PROCESSED_DIR / "team_features_sample.parquet", index=False)
    logger.info(f"  Saved processed data to {PROCESSED_DIR}")

    # Step 5: Train model and generate report
    logger.info("\n[Step 5/5] Training model and generating report...")
    metrics = train_simple_model(game_features_df)

    report_path = Path(__file__).parent.parent.parent / "docs" / "reports" / "sample_phase1.md"
    generate_sample_report(metrics, report_path)

    logger.info("\n" + "=" * 60)
    logger.info("Sample Phase 1 Pipeline Complete!")
    logger.info("=" * 60)

    return {
        "n_games": len(games_df),
        "n_features": len(game_features_df.columns),
        "metrics": metrics,
        "artifacts": {
            "game_features": str(PROCESSED_DIR / "game_features_sample.parquet"),
            "team_features": str(PROCESSED_DIR / "team_features_sample.parquet"),
            "report": str(report_path),
        },
    }


if __name__ == "__main__":
    results = run_sample_pipeline()
    print(f"\nPipeline completed successfully!")
    print(f"  Games processed: {results['n_games']}")
    print(f"  Feature columns: {results['n_features']}")
    print(f"  Test accuracy: {results['metrics']['accuracy']:.4f}")
