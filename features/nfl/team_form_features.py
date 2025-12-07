"""
NFL Team Form Features

Computes rolling team performance metrics (win rates, point differentials, etc.)
for use in baseline models.

CRITICAL: Ensures no data leakage by excluding the current game from rolling windows.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import logging
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_win_loss(team_stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate win/loss for each team-game.
    
    Args:
        team_stats_df: DataFrame with team stats (points_for, points_against)
    
    Returns:
        DataFrame with 'win' column added (1 for win, 0 for loss, 0.5 for tie)
    """
    df = team_stats_df.copy()
    df["win"] = (df["points_for"] > df["points_against"]).astype(int)
    df.loc[df["points_for"] == df["points_against"], "win"] = 0.5  # Ties
    df["win"] = df["win"].astype(float)  # Convert to float to allow 0.5
    return df


def calculate_point_differential(team_stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate point differential for each team-game.
    
    Args:
        team_stats_df: DataFrame with team stats
    
    Returns:
        DataFrame with 'point_diff' column added
    """
    df = team_stats_df.copy()
    df["point_diff"] = df["points_for"] - df["points_against"]
    return df


def calculate_turnover_differential(team_stats_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate turnover differential for each team-game.
    
    For each game, we need to know both teams' turnovers to calculate differential.
    
    Args:
        team_stats_df: DataFrame with team stats (may have turnovers column)
        games_df: DataFrame with game-level data (for matching opponent turnovers)
    
    Returns:
        DataFrame with 'turnover_diff' column added
    """
    df = team_stats_df.copy()
    
    # If turnovers are available in team_stats_df, calculate differential
    if "turnovers" in df.columns:
        # Merge with opponent turnovers
        # Create opponent stats lookup
        opponent_stats = df.merge(
            df[["game_id", "team", "turnovers"]],
            on="game_id",
            suffixes=("", "_opp"),
        )
        opponent_stats = opponent_stats[opponent_stats["team"] != opponent_stats["team_opp"]]
        
        # Merge back to get opponent turnovers
        df = df.merge(
            opponent_stats[["game_id", "team", "turnovers_opp"]],
            on=["game_id", "team"],
            how="left",
        )
        df["turnover_diff"] = df["turnovers"] - df.get("turnovers_opp", 0)
    else:
        # No turnover data available - set to 0
        df["turnover_diff"] = 0
    
    return df


def compute_rolling_features(
    team_stats_df: pd.DataFrame,
    windows: list = [4, 8, 16],
) -> pd.DataFrame:
    """
    Compute rolling features for each team.
    
    CRITICAL: Excludes the current game from rolling windows to prevent data leakage.
    
    Args:
        team_stats_df: DataFrame with team stats, must be sorted chronologically
        windows: List of window sizes for rolling calculations
    
    Returns:
        DataFrame with rolling features added
    """
    logger.info("Computing rolling features (excluding current game to prevent leakage)")
    
    # Ensure we have required columns
    required_cols = ["game_id", "team", "points_for", "points_against", "win", "point_diff"]
    missing_cols = [col for col in required_cols if col not in team_stats_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Sort by team and date (we'll need date from games)
    # For now, sort by game_id (which includes season/week)
    df = team_stats_df.copy()
    
    # Add a numeric ordering based on game_id (season_week)
    # Extract season and week for proper ordering
    extracted = df["game_id"].str.extract(r"nfl_(\d{4})_(\d{2})")
    df["season"] = pd.to_numeric(extracted[0], errors="coerce")
    df["week"] = pd.to_numeric(extracted[1], errors="coerce")
    
    # Create a sortable game number (season * 100 + week)
    df["game_number"] = df["season"] * 100 + df["week"]
    df = df.sort_values(["team", "game_number"]).reset_index(drop=True)
    
    # Group by team and compute rolling features
    result_rows = []
    
    for team in df["team"].unique():
        team_df = df[df["team"] == team].copy()
        team_df = team_df.sort_values("game_number").reset_index(drop=True)
        
        for idx, row in team_df.iterrows():
            # Get historical games (before current game)
            # CRITICAL: Use idx (not idx+1) to exclude current game
            historical = team_df.iloc[:idx]  # Excludes current game
            
            feature_row = row.to_dict()
            
            # Compute rolling features for each window
            for window in windows:
                # Get last N games (excluding current)
                window_data = historical.tail(window)
                
                if len(window_data) > 0:
                    # Win rate
                    feature_row[f"win_rate_last{window}"] = window_data["win"].mean()
                    
                    # Point differential average
                    feature_row[f"pdiff_last{window}"] = window_data["point_diff"].mean()
                    
                    # Points for average
                    feature_row[f"points_for_last{window}"] = window_data["points_for"].mean()
                    
                    # Points against average
                    feature_row[f"points_against_last{window}"] = window_data["points_against"].mean()
                    
                    # Turnover differential average (if available)
                    if "turnover_diff" in window_data.columns:
                        feature_row[f"turnover_diff_last{window}"] = window_data["turnover_diff"].mean()
                    else:
                        feature_row[f"turnover_diff_last{window}"] = 0.0
                else:
                    # Not enough history - set to NaN or 0
                    feature_row[f"win_rate_last{window}"] = 0.0
                    feature_row[f"pdiff_last{window}"] = 0.0
                    feature_row[f"points_for_last{window}"] = 0.0
                    feature_row[f"points_against_last{window}"] = 0.0
                    feature_row[f"turnover_diff_last{window}"] = 0.0
            
            result_rows.append(feature_row)
    
    result_df = pd.DataFrame(result_rows)
    
    # Clean up temporary columns
    result_df = result_df.drop(columns=["game_number"], errors="ignore")
    # Keep season and week as they might be useful
    
    logger.info(f"Computed rolling features for {len(result_df)} team-games")
    return result_df


def generate_team_form_features(
    team_stats_path: Optional[Path] = None,
    games_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Main function to generate team form features.
    
    Args:
        team_stats_path: Path to team_stats.parquet. If None, uses default.
        games_path: Path to games.parquet (for date ordering). If None, uses default.
        output_path: Path to save features. If None, uses default.
    
    Returns:
        DataFrame with team form features
    """
    if team_stats_path is None:
        team_stats_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "nfl"
            / "staged"
            / "team_stats.parquet"
        )
    
    if games_path is None:
        games_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "nfl"
            / "staged"
            / "games.parquet"
        )
    
    if output_path is None:
        output_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "nfl"
            / "processed"
            / "team_baseline_features.parquet"
        )
    
    # Check if output exists and is newer than inputs (simple caching)
    if output_path.exists():
        output_mtime = os.path.getmtime(output_path)
        team_stats_mtime = os.path.getmtime(team_stats_path) if team_stats_path.exists() else 0
        games_mtime = os.path.getmtime(games_path) if games_path.exists() else 0
        
        if output_mtime >= team_stats_mtime and output_mtime >= games_mtime:
            logger.info(f"✓ Feature table exists and is up-to-date: {output_path}")
            logger.info("  Skipping recomputation (using cached features)")
            return pd.read_parquet(output_path)
        else:
            logger.info(f"Feature table exists but inputs have changed, recomputing...")
    
    # Load data
    logger.info(f"Loading team stats from {team_stats_path}")
    team_stats_df = pd.read_parquet(team_stats_path)
    
    logger.info(f"Loading games from {games_path}")
    games_df = pd.read_parquet(games_path)
    
    # Merge with games to get date for proper ordering
    team_stats_df = team_stats_df.merge(
        games_df[["game_id", "date"]],
        on="game_id",
        how="left",
    )
    
    # Calculate derived metrics
    team_stats_df = calculate_win_loss(team_stats_df)
    team_stats_df = calculate_point_differential(team_stats_df)
    team_stats_df = calculate_turnover_differential(team_stats_df, games_df)
    
    # Sort by date for proper chronological ordering
    team_stats_df = team_stats_df.sort_values(["team", "date"]).reset_index(drop=True)
    
    # Compute rolling features
    features_df = compute_rolling_features(team_stats_df, windows=[4, 8, 16])
    
    # Select final columns
    feature_cols = [
        "game_id",
        "team",
        "is_home",
        "win_rate_last4",
        "win_rate_last8",
        "win_rate_last16",
        "pdiff_last4",
        "pdiff_last8",
        "pdiff_last16",
        "points_for_last4",
        "points_for_last8",
        "points_for_last16",
        "points_against_last4",
        "points_against_last8",
        "points_against_last16",
        "turnover_diff_last4",
        "turnover_diff_last8",
        "turnover_diff_last16",
    ]
    
    # Ensure all columns exist
    available_cols = [col for col in feature_cols if col in features_df.columns]
    final_df = features_df[available_cols].copy()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_path, index=False)
    logger.info(f"Saved team form features to {output_path}")
    
    return final_df


if __name__ == "__main__":
    df = generate_team_form_features()
    print(f"\nGenerated features for {len(df)} team-games")
    print(f"\nFeature columns: {df.columns.tolist()}")
    print(f"\nSample features:")
    print(df.head(10))

