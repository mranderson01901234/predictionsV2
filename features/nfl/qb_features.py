"""
NFL QB-Level Features

Computes quarterback-level metrics per game from play-by-play data.
Identifies the starting QB (most dropbacks) and computes EPA, CPOE, and other metrics.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def identify_starting_qb(plays_df: pd.DataFrame, game_id: str, team: str) -> Optional[str]:
    """
    Identify the starting QB for a team in a game.
    
    Heuristic: QB with the most dropbacks (pass attempts + sacks + scrambles).
    
    Args:
        plays_df: Play-by-play DataFrame
        game_id: Game identifier
        team: Team abbreviation
    
    Returns:
        QB player ID or name, or None if no QB found
    """
    # Filter to plays for this team in this game
    team_plays = plays_df[
        (plays_df["game_id"] == game_id) &
        (plays_df["posteam"] == team) &
        (plays_df["is_pass"] == 1)  # Pass plays only
    ]
    
    if len(team_plays) == 0:
        return None
    
    # Try to get passer_id column (nflverse standard)
    qb_col = None
    for col in ["passer_id", "passer_player_id", "passer", "qb_id"]:
        if col in team_plays.columns:
            qb_col = col
            break
    
    if qb_col is None:
        logger.warning(f"No QB identifier column found for {game_id} {team}")
        return None
    
    # Count dropbacks per QB
    # Dropbacks = pass attempts + sacks + scrambles
    # For simplicity, use pass attempts + sacks (if available)
    team_plays = team_plays.copy()  # Avoid SettingWithCopyWarning
    if "sack" in team_plays.columns:
        # Include sacks in dropback count
        team_plays["_dropbacks"] = team_plays["is_pass"] + team_plays["sack"].fillna(0)
    else:
        team_plays["_dropbacks"] = team_plays["is_pass"]
    
    # Group by QB and count dropbacks
    qb_dropbacks = team_plays.groupby(qb_col)["_dropbacks"].sum().sort_values(ascending=False)
    
    if len(qb_dropbacks) == 0:
        return None
    
    # Return QB with most dropbacks
    starting_qb = qb_dropbacks.index[0]
    return starting_qb


def compute_qb_metrics(
    plays_df: pd.DataFrame,
    game_id: str,
    team: str,
    qb_id: str,
) -> dict:
    """
    Compute QB metrics for a specific QB in a game.
    
    Args:
        plays_df: Play-by-play DataFrame
        game_id: Game identifier
        team: Team abbreviation
        qb_id: QB identifier
    
    Returns:
        Dictionary with QB metrics
    """
    # Filter to plays for this QB
    qb_col = None
    for col in ["passer_id", "passer_player_id", "passer", "qb_id"]:
        if col in plays_df.columns:
            qb_col = col
            break
    
    if qb_col is None:
        return {}
    
    qb_plays = plays_df[
        (plays_df["game_id"] == game_id) &
        (plays_df["posteam"] == team) &
        (plays_df[qb_col] == qb_id) &
        (plays_df["is_pass"] == 1)
    ]
    
    if len(qb_plays) == 0:
        return {}
    
    # Count dropbacks
    if "sack" in qb_plays.columns:
        dropbacks = (qb_plays["is_pass"] + qb_plays["sack"].fillna(0)).sum()
    else:
        dropbacks = qb_plays["is_pass"].sum()
    
    if dropbacks == 0:
        return {}
    
    # EPA per dropback
    epa_per_dropback = qb_plays["epa"].sum() / dropbacks
    
    # Success rate
    success_rate = qb_plays["success"].mean() if len(qb_plays) > 0 else 0.0
    
    # Sack rate
    sacks = qb_plays.get("sack", pd.Series([0] * len(qb_plays))).fillna(0).sum()
    sack_rate = sacks / dropbacks if dropbacks > 0 else 0.0
    
    # INT rate
    interceptions = qb_plays.get("interception", qb_plays.get("int", pd.Series([0] * len(qb_plays)))).fillna(0).sum()
    attempts = qb_plays["is_pass"].sum()
    int_rate = interceptions / attempts if attempts > 0 else 0.0
    
    # CPOE (Completion Percentage Over Expected) - if available
    cpoe = None
    if "cpoe" in qb_plays.columns:
        cpoe = qb_plays["cpoe"].mean()
    elif "cpoe_total" in qb_plays.columns:
        cpoe = qb_plays["cpoe_total"].iloc[0] if len(qb_plays) > 0 else None
    
    # Air yards per attempt - if available
    air_yards_per_attempt = None
    if "air_yards" in qb_plays.columns:
        air_yards = qb_plays["air_yards"].fillna(0).sum()
        air_yards_per_attempt = air_yards / attempts if attempts > 0 else 0.0
    
    return {
        "qb_id": qb_id,
        "qb_epa_per_dropback": epa_per_dropback,
        "qb_cpoe": cpoe,
        "qb_air_yards_per_attempt": air_yards_per_attempt,
        "qb_sack_rate": sack_rate,
        "qb_int_rate": int_rate,
        "qb_success_rate": success_rate,
        "qb_dropbacks": dropbacks,
    }


def compute_team_qb_features(
    plays_df: pd.DataFrame,
    games_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute QB features for each team in each game.
    
    Args:
        plays_df: Play-by-play DataFrame
        games_df: Optional games DataFrame for validation
    
    Returns:
        DataFrame with one row per team per game with QB metrics
    """
    logger.info("Computing QB features")
    
    # Get unique team-game combinations (include all teams, not just those with pass plays)
    team_games = plays_df[
        (plays_df["posteam"].notna())
    ][["game_id", "posteam"]].drop_duplicates()
    
    result_rows = []
    
    for _, row in team_games.iterrows():
        game_id = row["game_id"]
        team = row["posteam"]
        
        # Check if team has any pass plays
        team_has_passes = plays_df[
            (plays_df["game_id"] == game_id) &
            (plays_df["posteam"] == team) &
            (plays_df["is_pass"] == 1)
        ]
        
        if len(team_has_passes) == 0:
            # No pass plays - create row with null QB metrics
            result_rows.append({
                "game_id": game_id,
                "team": team,
                "qb_id": None,
                "qb_epa_per_dropback": None,
                "qb_cpoe": None,
                "qb_air_yards_per_attempt": None,
                "qb_sack_rate": None,
                "qb_int_rate": None,
                "qb_success_rate": None,
                "qb_dropbacks": 0,
            })
            continue
        
        # Identify starting QB
        qb_id = identify_starting_qb(plays_df, game_id, team)
        
        if qb_id is None:
            logger.warning(f"No QB found for {team} in {game_id}")
            # Create row with null QB metrics
            result_rows.append({
                "game_id": game_id,
                "team": team,
                "qb_id": None,
                "qb_epa_per_dropback": None,
                "qb_cpoe": None,
                "qb_air_yards_per_attempt": None,
                "qb_sack_rate": None,
                "qb_int_rate": None,
                "qb_success_rate": None,
                "qb_dropbacks": 0,
            })
            continue
        
        # Compute QB metrics
        qb_metrics = compute_qb_metrics(plays_df, game_id, team, qb_id)
        
        result_rows.append({
            "game_id": game_id,
            "team": team,
            **qb_metrics,
        })
    
    result_df = pd.DataFrame(result_rows)
    
    # Handle empty result_df
    if len(result_df) == 0:
        # Return empty DataFrame with correct columns
        result_df = pd.DataFrame(columns=[
            "game_id", "team", "qb_id", "qb_epa_per_dropback", "qb_cpoe",
            "qb_air_yards_per_attempt", "qb_sack_rate", "qb_int_rate",
            "qb_success_rate", "qb_dropbacks"
        ])
    
    # Validate ranges (only if we have data)
    if len(result_df) > 0:
        # EPA per dropback: typically -0.5 to +0.5
        # Sack rate: 0 to 1
        # INT rate: 0 to 1
        # Success rate: 0 to 1
        
        if "qb_epa_per_dropback" in result_df.columns:
            invalid_epa = result_df[
                (result_df["qb_epa_per_dropback"].notna()) &
                ((result_df["qb_epa_per_dropback"] < -2.0) | (result_df["qb_epa_per_dropback"] > 2.0))
            ]
            if len(invalid_epa) > 0:
                logger.warning(f"{len(invalid_epa)} team-games with unusual QB EPA values")
        
        if "qb_sack_rate" in result_df.columns:
            invalid_sack_rate = result_df[
                (result_df["qb_sack_rate"].notna()) &
                ((result_df["qb_sack_rate"] < 0) | (result_df["qb_sack_rate"] > 1))
            ]
            if len(invalid_sack_rate) > 0:
                logger.warning(f"{len(invalid_sack_rate)} team-games with invalid sack rate")
        
        if "qb_int_rate" in result_df.columns:
            invalid_int_rate = result_df[
                (result_df["qb_int_rate"].notna()) &
                ((result_df["qb_int_rate"] < 0) | (result_df["qb_int_rate"] > 1))
            ]
            if len(invalid_int_rate) > 0:
                logger.warning(f"{len(invalid_int_rate)} team-games with invalid INT rate")
    
    logger.info(f"Computed QB features for {len(result_df)} team-games")
    return result_df


def generate_team_qb_features(
    plays_path: Optional[Path] = None,
    games_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Main function to generate QB features.
    
    Args:
        plays_path: Path to plays.parquet. If None, uses default.
        games_path: Path to games.parquet (for validation). If None, uses default.
        output_path: Path to save features. If None, uses default.
    
    Returns:
        DataFrame with QB features
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
            / "team_qb_features.parquet"
        )
    
    # Load data
    logger.info(f"Loading plays from {plays_path}")
    plays_df = pd.read_parquet(plays_path)
    
    games_df = None
    if games_path.exists():
        logger.info(f"Loading games from {games_path}")
        games_df = pd.read_parquet(games_path)
    
    # Compute QB features
    qb_features_df = compute_team_qb_features(plays_df, games_df)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    qb_features_df.to_parquet(output_path, index=False)
    logger.info(f"Saved QB features to {output_path}")
    
    return qb_features_df


if __name__ == "__main__":
    df = generate_team_qb_features()
    print(f"\nGenerated QB features for {len(df)} team-games")
    print(f"\nFeature columns: {df.columns.tolist()}")
    print(f"\nSample features:")
    print(df.head(10))

