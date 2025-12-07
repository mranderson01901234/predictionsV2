"""
NFL Rolling EPA and Success Rate Features

Computes rolling windows of EPA and success rate metrics for each team,
excluding the current game to prevent data leakage.

Windows: last3, last5, last8 games
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_rolling_epa_features(
    team_epa_df: pd.DataFrame,
    games_df: Optional[pd.DataFrame] = None,
    windows: list = [3, 5, 8],
) -> pd.DataFrame:
    """
    Compute rolling EPA and success rate features for each team.
    
    CRITICAL: Excludes the current game from rolling windows to prevent data leakage.
    
    Args:
        team_epa_df: DataFrame with per-game EPA features (from epa_features.py)
        games_df: Optional games DataFrame for date ordering
        windows: List of window sizes (default: [3, 5, 8])
    
    Returns:
        DataFrame with rolling EPA features added
    """
    logger.info("Computing rolling EPA features (excluding current game to prevent leakage)")
    
    # Required columns from team_epa_features
    required_cols = [
        "game_id",
        "team",
        "offensive_epa_per_play",
        "offensive_pass_epa",
        "offensive_run_epa",
        "offensive_success_rate",
        "offensive_pass_success_rate",
        "offensive_run_success_rate",
        "defensive_epa_per_play_allowed",
        "defensive_pass_epa_allowed",
        "defensive_run_epa_allowed",
        "defensive_success_rate_allowed",
        "defensive_pass_success_rate_allowed",
        "defensive_run_success_rate_allowed",
    ]
    
    missing_cols = [col for col in required_cols if col not in team_epa_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df = team_epa_df.copy()
    
    # Merge with games to get date for proper ordering
    if games_df is None:
        games_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "nfl"
            / "staged"
            / "games.parquet"
        )
        if games_path.exists():
            games_df = pd.read_parquet(games_path)
        else:
            logger.warning("games.parquet not found, using game_id for ordering")
            games_df = None
    
    if games_df is not None:
        df = df.merge(
            games_df[["game_id", "date", "season", "week"]],
            on="game_id",
            how="left",
        )
        # Sort by team and date
        df = df.sort_values(["team", "date"]).reset_index(drop=True)
    else:
        # Fallback: extract season/week from game_id
        extracted = df["game_id"].str.extract(r"nfl_(\d{4})_(\d{2})")
        df["season"] = pd.to_numeric(extracted[0], errors="coerce")
        df["week"] = pd.to_numeric(extracted[1], errors="coerce")
        df["game_number"] = df["season"] * 100 + df["week"]
        df = df.sort_values(["team", "game_number"]).reset_index(drop=True)
    
    # Group by team and compute rolling features
    result_rows = []
    
    for team in df["team"].unique():
        team_df = df[df["team"] == team].copy()
        team_df = team_df.reset_index(drop=True)
        
        for idx, row in team_df.iterrows():
            # Get historical games (before current game)
            # CRITICAL: Use idx (not idx+1) to exclude current game
            historical = team_df.iloc[:idx]  # Excludes current game
            
            feature_row = row.to_dict()
            
            # Compute rolling features for each window
            for window in windows:
                # Get last N games (excluding current)
                window_data = historical.tail(window)
                
                if len(window_data) >= window:
                    # Full window available
                    # Offensive metrics
                    feature_row[f"off_epa_last{window}"] = window_data["offensive_epa_per_play"].mean()
                    feature_row[f"off_pass_epa_last{window}"] = window_data["offensive_pass_epa"].mean()
                    feature_row[f"off_run_epa_last{window}"] = window_data["offensive_run_epa"].mean()
                    feature_row[f"off_sr_last{window}"] = window_data["offensive_success_rate"].mean()
                    feature_row[f"off_pass_sr_last{window}"] = window_data["offensive_pass_success_rate"].mean()
                    feature_row[f"off_run_sr_last{window}"] = window_data["offensive_run_success_rate"].mean()
                    
                    # Defensive metrics
                    feature_row[f"def_epa_allowed_last{window}"] = window_data["defensive_epa_per_play_allowed"].mean()
                    feature_row[f"def_pass_epa_allowed_last{window}"] = window_data["defensive_pass_epa_allowed"].mean()
                    feature_row[f"def_run_epa_allowed_last{window}"] = window_data["defensive_run_epa_allowed"].mean()
                    feature_row[f"def_sr_allowed_last{window}"] = window_data["defensive_success_rate_allowed"].mean()
                    feature_row[f"def_pass_sr_allowed_last{window}"] = window_data["defensive_pass_success_rate_allowed"].mean()
                    feature_row[f"def_run_sr_allowed_last{window}"] = window_data["defensive_run_success_rate_allowed"].mean()
                elif len(window_data) > 0:
                    # Partial window - use available games
                    # This handles early-season games
                    feature_row[f"off_epa_last{window}"] = window_data["offensive_epa_per_play"].mean()
                    feature_row[f"off_pass_epa_last{window}"] = window_data["offensive_pass_epa"].mean()
                    feature_row[f"off_run_epa_last{window}"] = window_data["offensive_run_epa"].mean()
                    feature_row[f"off_sr_last{window}"] = window_data["offensive_success_rate"].mean()
                    feature_row[f"off_pass_sr_last{window}"] = window_data["offensive_pass_success_rate"].mean()
                    feature_row[f"off_run_sr_last{window}"] = window_data["offensive_run_success_rate"].mean()
                    
                    feature_row[f"def_epa_allowed_last{window}"] = window_data["defensive_epa_per_play_allowed"].mean()
                    feature_row[f"def_pass_epa_allowed_last{window}"] = window_data["defensive_pass_epa_allowed"].mean()
                    feature_row[f"def_run_epa_allowed_last{window}"] = window_data["defensive_run_epa_allowed"].mean()
                    feature_row[f"def_sr_allowed_last{window}"] = window_data["defensive_success_rate_allowed"].mean()
                    feature_row[f"def_pass_sr_allowed_last{window}"] = window_data["defensive_pass_success_rate_allowed"].mean()
                    feature_row[f"def_run_sr_allowed_last{window}"] = window_data["defensive_run_success_rate_allowed"].mean()
                else:
                    # No history - set to NaN (will be handled later)
                    feature_row[f"off_epa_last{window}"] = None
                    feature_row[f"off_pass_epa_last{window}"] = None
                    feature_row[f"off_run_epa_last{window}"] = None
                    feature_row[f"off_sr_last{window}"] = None
                    feature_row[f"off_pass_sr_last{window}"] = None
                    feature_row[f"off_run_sr_last{window}"] = None
                    
                    feature_row[f"def_epa_allowed_last{window}"] = None
                    feature_row[f"def_pass_epa_allowed_last{window}"] = None
                    feature_row[f"def_run_epa_allowed_last{window}"] = None
                    feature_row[f"def_sr_allowed_last{window}"] = None
                    feature_row[f"def_pass_sr_allowed_last{window}"] = None
                    feature_row[f"def_run_sr_allowed_last{window}"] = None
            
            result_rows.append(feature_row)
    
    result_df = pd.DataFrame(result_rows)
    
    # Clean up temporary columns
    result_df = result_df.drop(columns=["game_number"], errors="ignore")
    
    logger.info(f"Computed rolling EPA features for {len(result_df)} team-games")
    return result_df


def generate_rolling_epa_features(
    team_epa_features_path: Optional[Path] = None,
    games_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    windows: list = [3, 5, 8],
) -> pd.DataFrame:
    """
    Main function to generate rolling EPA features.
    
    Args:
        team_epa_features_path: Path to team_epa_features.parquet. If None, uses default.
        games_path: Path to games.parquet (for date ordering). If None, uses default.
        output_path: Path to save features. If None, uses default.
        windows: List of window sizes for rolling calculations
    
    Returns:
        DataFrame with rolling EPA features
    """
    if team_epa_features_path is None:
        team_epa_features_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "nfl"
            / "processed"
            / "team_epa_features.parquet"
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
            / "team_rolling_epa_features.parquet"
        )
    
    # Load data
    logger.info(f"Loading team EPA features from {team_epa_features_path}")
    team_epa_df = pd.read_parquet(team_epa_features_path)
    
    games_df = None
    if games_path.exists():
        logger.info(f"Loading games from {games_path}")
        games_df = pd.read_parquet(games_path)
    
    # Compute rolling features
    rolling_df = compute_rolling_epa_features(team_epa_df, games_df, windows)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rolling_df.to_parquet(output_path, index=False)
    logger.info(f"Saved rolling EPA features to {output_path}")
    
    return rolling_df


if __name__ == "__main__":
    df = generate_rolling_epa_features()
    print(f"\nGenerated rolling EPA features for {len(df)} team-games")
    print(f"\nRolling feature columns: {[col for col in df.columns if 'last' in col]}")
    print(f"\nSample features:")
    print(df[["game_id", "team", "off_epa_last3", "off_epa_last5", "def_epa_allowed_last3"]].head(10))

