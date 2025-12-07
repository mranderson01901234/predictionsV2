"""
NFL EPA and Success Rate Features

Computes team-level EPA and success rate metrics from play-by-play data
for use in advanced models.

Metrics computed:
- Offensive EPA/play (overall, pass, run)
- Defensive EPA/play allowed (overall, pass, run)
- Success rates (offensive, defensive, pass, run)
- Situational splits (3rd down, red zone, late downs, early downs)
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_team_epa_features(
    plays_df: pd.DataFrame,
    games_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute EPA and success rate features for each team in each game.
    
    Args:
        plays_df: DataFrame with play-by-play data (PlayByPlay schema)
        games_df: Optional games DataFrame for validation
    
    Returns:
        DataFrame with one row per team per game with EPA metrics
    """
    logger.info("Computing team EPA features")
    
    # Filter to valid offensive plays (has posteam, EPA)
    offensive_plays = plays_df[
        (plays_df["posteam"].notna()) &
        (plays_df["epa"].notna()) &
        (plays_df["is_pass"] + plays_df["is_run"] > 0)  # Must be pass or run
    ].copy()
    
    if len(offensive_plays) == 0:
        raise ValueError("No valid offensive plays found")
    
    logger.info(f"Processing {len(offensive_plays)} offensive plays")
    
    # Group by game_id and posteam (offensive team)
    result_rows = []
    
    for (game_id, team), group in offensive_plays.groupby(["game_id", "posteam"]):
        # Overall offensive metrics
        total_plays = len(group)
        if total_plays == 0:
            continue
        
        offensive_epa_per_play = group["epa"].mean()
        offensive_success_rate = group["success"].mean()
        
        # Pass plays
        pass_plays = group[group["is_pass"] == 1]
        offensive_pass_epa = pass_plays["epa"].mean() if len(pass_plays) > 0 else 0.0
        offensive_pass_success_rate = pass_plays["success"].mean() if len(pass_plays) > 0 else 0.0
        
        # Run plays
        run_plays = group[group["is_run"] == 1]
        offensive_run_epa = run_plays["epa"].mean() if len(run_plays) > 0 else 0.0
        offensive_run_success_rate = run_plays["success"].mean() if len(run_plays) > 0 else 0.0
        
        # Situational splits
        # Third down
        third_down = group[group["down"] == 3]
        third_down_epa = third_down["epa"].mean() if len(third_down) > 0 else 0.0
        
        # Red zone (yardline_100 <= 20)
        red_zone = group[group["yardline_100"] <= 20]
        red_zone_epa = red_zone["epa"].mean() if len(red_zone) > 0 else 0.0
        
        # Late downs (3rd and 4th)
        late_downs = group[group["down"].isin([3, 4])]
        late_down_epa = late_downs["epa"].mean() if len(late_downs) > 0 else 0.0
        
        # Early downs (1st and 2nd)
        early_downs = group[group["down"].isin([1, 2])]
        early_down_epa = early_downs["epa"].mean() if len(early_downs) > 0 else 0.0
        
        result_rows.append({
            "game_id": game_id,
            "team": team,
            "offensive_epa_per_play": offensive_epa_per_play,
            "offensive_pass_epa": offensive_pass_epa,
            "offensive_run_epa": offensive_run_epa,
            "offensive_success_rate": offensive_success_rate,
            "offensive_pass_success_rate": offensive_pass_success_rate,
            "offensive_run_success_rate": offensive_run_success_rate,
            "third_down_epa": third_down_epa,
            "red_zone_epa": red_zone_epa,
            "late_down_epa": late_down_epa,
            "early_down_epa": early_down_epa,
        })
    
    offensive_df = pd.DataFrame(result_rows)
    logger.info(f"Computed offensive metrics for {len(offensive_df)} team-games")
    
    # Now compute defensive metrics (flip posteam/defteam)
    defensive_plays = plays_df[
        (plays_df["defteam"].notna()) &
        (plays_df["epa"].notna()) &
        (plays_df["is_pass"] + plays_df["is_run"] > 0)
    ].copy()
    
    # For defense, EPA allowed is from the defense's perspective
    # If offense has positive EPA, defense allowed positive EPA (bad defense)
    # If offense has negative EPA, defense allowed negative EPA (good defense)
    # Standard convention: defensive EPA allowed = opponent offensive EPA (no negation)
    # But we want negative = good defense, so we negate
    defensive_plays["epa_allowed"] = -defensive_plays["epa"]
    
    defensive_result_rows = []
    
    for (game_id, team), group in defensive_plays.groupby(["game_id", "defteam"]):
        # Overall defensive metrics
        total_plays = len(group)
        if total_plays == 0:
            continue
        
        # Defensive EPA allowed (negative EPA = good defense)
        defensive_epa_per_play_allowed = group["epa_allowed"].mean()
        defensive_success_rate_allowed = 1.0 - group["success"].mean()  # Flip success rate
        
        # Pass defense
        pass_plays = group[group["is_pass"] == 1]
        defensive_pass_epa_allowed = pass_plays["epa_allowed"].mean() if len(pass_plays) > 0 else 0.0
        defensive_pass_success_rate_allowed = (
            1.0 - pass_plays["success"].mean() if len(pass_plays) > 0 else 0.0
        )
        
        # Run defense
        run_plays = group[group["is_run"] == 1]
        defensive_run_epa_allowed = run_plays["epa_allowed"].mean() if len(run_plays) > 0 else 0.0
        defensive_run_success_rate_allowed = (
            1.0 - run_plays["success"].mean() if len(run_plays) > 0 else 0.0
        )
        
        defensive_result_rows.append({
            "game_id": game_id,
            "team": team,
            "defensive_epa_per_play_allowed": defensive_epa_per_play_allowed,
            "defensive_pass_epa_allowed": defensive_pass_epa_allowed,
            "defensive_run_epa_allowed": defensive_run_epa_allowed,
            "defensive_success_rate_allowed": defensive_success_rate_allowed,
            "defensive_pass_success_rate_allowed": defensive_pass_success_rate_allowed,
            "defensive_run_success_rate_allowed": defensive_run_success_rate_allowed,
        })
    
    defensive_df = pd.DataFrame(defensive_result_rows)
    logger.info(f"Computed defensive metrics for {len(defensive_df)} team-games")
    
    # Merge offensive and defensive metrics
    result_df = offensive_df.merge(
        defensive_df,
        on=["game_id", "team"],
        how="outer",
    )
    
    # Fill missing values with 0 (no plays in that category)
    result_df = result_df.fillna(0.0)
    
    # Validate ranges
    # Typical offensive EPA/play: -0.2 to +0.3
    # Pass EPA usually > run EPA
    # Defensive EPA allowed usually negative for strong defenses
    invalid_epa = result_df[
        (result_df["offensive_epa_per_play"] < -1.0) |
        (result_df["offensive_epa_per_play"] > 1.0)
    ]
    if len(invalid_epa) > 0:
        logger.warning(f"{len(invalid_epa)} team-games with unusual EPA values")
        logger.debug(f"Sample: {invalid_epa[['game_id', 'team', 'offensive_epa_per_play']].head()}")
    
    logger.info(f"Final EPA features: {len(result_df)} team-games")
    return result_df


def generate_team_epa_features(
    plays_path: Optional[Path] = None,
    games_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Main function to generate team EPA features.
    
    Args:
        plays_path: Path to plays.parquet. If None, uses default.
        games_path: Path to games.parquet (for validation). If None, uses default.
        output_path: Path to save features. If None, uses default.
    
    Returns:
        DataFrame with team EPA features
    """
    if plays_path is None:
        plays_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "nfl"
            / "staged"
            / "plays.parquet"
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
            / "team_epa_features.parquet"
        )
    
    # Load data
    logger.info(f"Loading plays from {plays_path}")
    plays_df = pd.read_parquet(plays_path)
    
    games_df = None
    if games_path.exists():
        logger.info(f"Loading games from {games_path}")
        games_df = pd.read_parquet(games_path)
    
    # Compute EPA features
    epa_features_df = compute_team_epa_features(plays_df, games_df)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epa_features_df.to_parquet(output_path, index=False)
    logger.info(f"Saved team EPA features to {output_path}")
    
    return epa_features_df


if __name__ == "__main__":
    df = generate_team_epa_features()
    print(f"\nGenerated EPA features for {len(df)} team-games")
    print(f"\nFeature columns: {df.columns.tolist()}")
    print(f"\nSample features:")
    print(df.head(10))

