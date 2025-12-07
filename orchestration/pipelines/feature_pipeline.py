"""
Feature Pipeline

Merges team form features with games and markets data to create
game-level feature tables for modeling.
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


def merge_team_features_to_games(
    games_markets_path: Optional[Path] = None,
    team_features_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Merge team form features into game-level feature table.
    
    Args:
        games_markets_path: Path to games_markets.parquet
        team_features_path: Path to team_baseline_features.parquet
        output_path: Path to save final features
    
    Returns:
        DataFrame with game-level features (home_* and away_* prefixed)
    """
    if games_markets_path is None:
        games_markets_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "nfl"
            / "staged"
            / "games_markets.parquet"
        )
    
    if team_features_path is None:
        team_features_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "nfl"
            / "processed"
            / "team_baseline_features.parquet"
        )
    
    if output_path is None:
        output_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "nfl"
            / "processed"
            / "game_features_baseline.parquet"
        )
    
    # Check if output exists and is newer than inputs (simple caching)
    if output_path.exists():
        output_mtime = os.path.getmtime(output_path)
        games_markets_mtime = os.path.getmtime(games_markets_path) if games_markets_path.exists() else 0
        team_features_mtime = os.path.getmtime(team_features_path) if team_features_path.exists() else 0
        
        if output_mtime >= games_markets_mtime and output_mtime >= team_features_mtime:
            logger.info(f"✓ Game features table exists and is up-to-date: {output_path}")
            logger.info("  Skipping recomputation (using cached features)")
            return pd.read_parquet(output_path)
        else:
            logger.info(f"Game features table exists but inputs have changed, recomputing...")
    
    # Load data
    logger.info(f"Loading games_markets from {games_markets_path}")
    games_markets_df = pd.read_parquet(games_markets_path)
    
    logger.info(f"Loading team features from {team_features_path}")
    team_features_df = pd.read_parquet(team_features_path)
    
    # Separate home and away team features
    home_features = team_features_df[team_features_df["is_home"] == True].copy()
    away_features = team_features_df[team_features_df["is_home"] == False].copy()
    
    # Prefix feature columns (exclude game_id, team, is_home)
    feature_cols = [col for col in team_features_df.columns 
                   if col not in ["game_id", "team", "is_home"]]
    
    # Rename home team features
    home_feature_map = {col: f"home_{col}" for col in feature_cols}
    home_features = home_features.rename(columns=home_feature_map)
    
    # Rename away team features
    away_feature_map = {col: f"away_{col}" for col in feature_cols}
    away_features = away_features.rename(columns=away_feature_map)
    
    # Merge home team features
    logger.info("Merging home team features")
    result_df = games_markets_df.merge(
        home_features[["game_id"] + [f"home_{col}" for col in feature_cols]],
        on="game_id",
        how="left",
    )
    
    # Merge away team features
    logger.info("Merging away team features")
    result_df = result_df.merge(
        away_features[["game_id"] + [f"away_{col}" for col in feature_cols]],
        on="game_id",
        how="left",
    )
    
    # Validate
    logger.info("Validating merged features")
    
    # Check for missing features
    missing_home = result_df[[f"home_{col}" for col in feature_cols]].isna().any(axis=1).sum()
    missing_away = result_df[[f"away_{col}" for col in feature_cols]].isna().any(axis=1).sum()
    
    if missing_home > 0:
        logger.warning(f"{missing_home} games missing home team features")
    if missing_away > 0:
        logger.warning(f"{missing_away} games missing away team features")
    
    # Check for duplicates
    if result_df["game_id"].duplicated().any():
        logger.error("Found duplicate game_ids in merged data!")
        raise ValueError("Duplicate game_ids found")
    
    # Remove rows with missing critical features
    initial_len = len(result_df)
    result_df = result_df.dropna(subset=[f"home_{feature_cols[0]}", f"away_{feature_cols[0]}"])
    if len(result_df) < initial_len:
        logger.warning(f"Dropped {initial_len - len(result_df)} rows with missing features")
    
    # Sort by season, week, date
    result_df = result_df.sort_values(["season", "week", "date"]).reset_index(drop=True)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(output_path, index=False)
    logger.info(f"Saved game-level features to {output_path}")
    logger.info(f"Final feature table: {len(result_df)} games, {len(result_df.columns)} columns")
    
    return result_df


def run_baseline_feature_pipeline() -> pd.DataFrame:
    """
    Run the complete baseline feature pipeline.
    
    Returns:
        DataFrame with game-level features
    """
    logger.info("=" * 60)
    logger.info("Running Baseline Feature Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Generate team form features (if not already done)
    from features.nfl.team_form_features import generate_team_form_features
    
    logger.info("\n[Step 1/2] Generating team form features...")
    team_features_df = generate_team_form_features()
    logger.info(f"✓ Team features ready: {len(team_features_df)} team-games")
    
    # Step 2: Merge into game-level features
    logger.info("\n[Step 2/2] Merging team features into game-level table...")
    game_features_df = merge_team_features_to_games()
    logger.info(f"✓ Game-level features ready: {len(game_features_df)} games")
    
    logger.info("\n" + "=" * 60)
    logger.info("Baseline Feature Pipeline Complete!")
    logger.info("=" * 60)
    
    return game_features_df


def merge_epa_features_to_games(
    games_markets_path: Optional[Path] = None,
    team_baseline_features_path: Optional[Path] = None,
    team_epa_features_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Merge baseline and EPA features into game-level feature table for Phase 2.
    
    Args:
        games_markets_path: Path to games_markets.parquet
        team_baseline_features_path: Path to team_baseline_features.parquet
        team_epa_features_path: Path to team_epa_features.parquet
        output_path: Path to save final features
    
    Returns:
        DataFrame with game-level features (home_* and away_* prefixed)
    """
    if games_markets_path is None:
        games_markets_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "nfl"
            / "staged"
            / "games_markets.parquet"
        )
    
    if team_baseline_features_path is None:
        team_baseline_features_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "nfl"
            / "processed"
            / "team_baseline_features.parquet"
        )
    
    if team_epa_features_path is None:
        team_epa_features_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "nfl"
            / "processed"
            / "team_epa_features.parquet"
        )
    
    if output_path is None:
        output_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "nfl"
            / "processed"
            / "game_features_phase2.parquet"
        )
    
    # Load data
    logger.info(f"Loading games_markets from {games_markets_path}")
    games_markets_df = pd.read_parquet(games_markets_path)
    
    logger.info(f"Loading baseline features from {team_baseline_features_path}")
    baseline_features_df = pd.read_parquet(team_baseline_features_path)
    
    logger.info(f"Loading EPA features from {team_epa_features_path}")
    epa_features_df = pd.read_parquet(team_epa_features_path)
    
    # Merge baseline features (home/away split)
    home_baseline = baseline_features_df[baseline_features_df["is_home"] == True].copy()
    away_baseline = baseline_features_df[baseline_features_df["is_home"] == False].copy()
    
    baseline_feature_cols = [
        col for col in baseline_features_df.columns
        if col not in ["game_id", "team", "is_home"]
    ]
    
    home_baseline_map = {col: f"home_{col}" for col in baseline_feature_cols}
    home_baseline = home_baseline.rename(columns=home_baseline_map)
    
    away_baseline_map = {col: f"away_{col}" for col in baseline_feature_cols}
    away_baseline = away_baseline.rename(columns=away_baseline_map)
    
    # Merge EPA features (need to match with home_team/away_team from games)
    # EPA features don't have is_home, so we need to match by team
    result_df = games_markets_df.copy()
    
    # Merge home baseline features
    logger.info("Merging home baseline features")
    result_df = result_df.merge(
        home_baseline[["game_id"] + [f"home_{col}" for col in baseline_feature_cols]],
        on="game_id",
        how="left",
    )
    
    # Merge away baseline features
    logger.info("Merging away baseline features")
    result_df = result_df.merge(
        away_baseline[["game_id"] + [f"away_{col}" for col in baseline_feature_cols]],
        on="game_id",
        how="left",
    )
    
    # Merge EPA features for home team
    logger.info("Merging home EPA features")
    home_epa = epa_features_df[
        epa_features_df["team"].isin(result_df["home_team"])
    ].copy()
    home_epa = home_epa.merge(
        result_df[["game_id", "home_team"]],
        left_on=["game_id", "team"],
        right_on=["game_id", "home_team"],
        how="inner",
    )
    
    epa_feature_cols = [
        col for col in epa_features_df.columns
        if col not in ["game_id", "team"]
    ]
    
    home_epa_map = {col: f"home_epa_{col}" for col in epa_feature_cols}
    home_epa = home_epa.rename(columns=home_epa_map)
    
    result_df = result_df.merge(
        home_epa[["game_id"] + [f"home_epa_{col}" for col in epa_feature_cols]],
        on="game_id",
        how="left",
    )
    
    # Merge EPA features for away team
    logger.info("Merging away EPA features")
    away_epa = epa_features_df[
        epa_features_df["team"].isin(result_df["away_team"])
    ].copy()
    away_epa = away_epa.merge(
        result_df[["game_id", "away_team"]],
        left_on=["game_id", "team"],
        right_on=["game_id", "away_team"],
        how="inner",
    )
    
    away_epa_map = {col: f"away_epa_{col}" for col in epa_feature_cols}
    away_epa = away_epa.rename(columns=away_epa_map)
    
    result_df = result_df.merge(
        away_epa[["game_id"] + [f"away_epa_{col}" for col in epa_feature_cols]],
        on="game_id",
        how="left",
    )
    
    # Validate
    logger.info("Validating merged features")
    
    # Check for missing EPA features
    epa_cols = [col for col in result_df.columns if "epa" in col.lower()]
    missing_epa = result_df[epa_cols].isna().any(axis=1).sum()
    if missing_epa > 0:
        logger.warning(f"{missing_epa} games missing EPA features")
    
    # Check for duplicates
    if result_df["game_id"].duplicated().any():
        logger.error("Found duplicate game_ids in merged data!")
        raise ValueError("Duplicate game_ids found")
    
    # Validate EPA ranges
    home_epa_cols = [col for col in epa_cols if col.startswith("home_epa_")]
    for col in home_epa_cols:
        if "epa_per_play" in col or "epa" in col:
            invalid = result_df[
                (result_df[col] < -1.0) | (result_df[col] > 1.0)
            ]
            if len(invalid) > 0:
                logger.warning(f"{len(invalid)} games with unusual {col} values")
    
    # Sort by season, week, date
    result_df = result_df.sort_values(["season", "week", "date"]).reset_index(drop=True)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(output_path, index=False)
    logger.info(f"Saved Phase 2 game-level features to {output_path}")
    logger.info(f"Final feature table: {len(result_df)} games, {len(result_df.columns)} columns")
    
    return result_df


def run_phase2_feature_pipeline() -> pd.DataFrame:
    """
    Run the complete Phase 2 feature pipeline (baseline + EPA).
    
    Returns:
        DataFrame with game-level features including EPA metrics
    """
    import time
    pipeline_start = time.time()
    
    logger.info("=" * 60)
    logger.info("Running Phase 2 Feature Pipeline (Baseline + EPA)")
    logger.info("=" * 60)
    
    # Step 1: Generate baseline features (if not already done)
    from features.nfl.team_form_features import generate_team_form_features
    
    step_start = time.time()
    logger.info("\n[Step 1/3] Generating baseline team form features...")
    baseline_features_df = generate_team_form_features()
    step_time = time.time() - step_start
    logger.info(f"✓ Generated baseline features: {len(baseline_features_df):,} team-games ({step_time:.2f}s)")
    
    # Step 2: Generate EPA features (if not already done)
    from features.nfl.epa_features import generate_team_epa_features
    
    step_start = time.time()
    logger.info("\n[Step 2/3] Generating EPA features...")
    epa_features_df = generate_team_epa_features()
    step_time = time.time() - step_start
    logger.info(f"✓ Generated EPA features: {len(epa_features_df):,} team-games ({step_time:.2f}s)")
    
    # Count EPA features added
    epa_feature_count = len([col for col in epa_features_df.columns if col not in ["game_id", "team"]])
    logger.info(f"  Added {epa_feature_count} EPA feature columns")
    
    # Step 3: Merge all features
    step_start = time.time()
    logger.info("\n[Step 3/3] Merging baseline + EPA features into game-level table...")
    game_features_df = merge_epa_features_to_games()
    step_time = time.time() - step_start
    logger.info(f"✓ Created Phase 2 game-level features: {len(game_features_df):,} games ({step_time:.2f}s)")
    
    # Count total features
    baseline_feature_count = len([col for col in game_features_df.columns 
                                  if col.startswith(("home_", "away_")) and "epa" not in col.lower()])
    total_feature_count = len([col for col in game_features_df.columns 
                              if col not in ["game_id", "season", "week", "date", "home_team", "away_team", 
                                            "home_score", "away_score", "close_spread", "close_total", 
                                            "open_spread", "open_total"]])
    logger.info(f"  Total features: {total_feature_count} (baseline: ~{baseline_feature_count}, EPA: ~{total_feature_count - baseline_feature_count})")
    
    pipeline_time = time.time() - pipeline_start
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2 Feature Pipeline Complete!")
    logger.info(f"Total pipeline time: {pipeline_time:.2f} seconds")
    logger.info("=" * 60)
    
    return game_features_df


def merge_phase2b_features(
    game_features_phase2_path: Optional[Path] = None,
    team_rolling_epa_path: Optional[Path] = None,
    team_qb_features_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Merge Phase 2B features (rolling EPA + QB) into Phase 2 feature table.
    
    Args:
        game_features_phase2_path: Path to game_features_phase2.parquet
        team_rolling_epa_path: Path to team_rolling_epa_features.parquet
        team_qb_features_path: Path to team_qb_features.parquet
        output_path: Path to save final features
    
    Returns:
        DataFrame with Phase 2B features (baseline + EPA + rolling EPA + QB)
    """
    if game_features_phase2_path is None:
        game_features_phase2_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "nfl"
            / "processed"
            / "game_features_phase2.parquet"
        )
    
    if team_rolling_epa_path is None:
        team_rolling_epa_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "nfl"
            / "processed"
            / "team_rolling_epa_features.parquet"
        )
    
    if team_qb_features_path is None:
        team_qb_features_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "nfl"
            / "processed"
            / "team_qb_features.parquet"
        )
    
    if output_path is None:
        output_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "nfl"
            / "processed"
            / "game_features_phase2b.parquet"
        )
    
    # Load data
    logger.info(f"Loading Phase 2 features from {game_features_phase2_path}")
    phase2_df = pd.read_parquet(game_features_phase2_path)
    
    logger.info(f"Loading rolling EPA features from {team_rolling_epa_path}")
    rolling_epa_df = pd.read_parquet(team_rolling_epa_path)
    
    logger.info(f"Loading QB features from {team_qb_features_path}")
    qb_features_df = pd.read_parquet(team_qb_features_path)
    
    result_df = phase2_df.copy()
    
    # Merge rolling EPA features for home team
    logger.info("Merging home rolling EPA features")
    home_rolling = rolling_epa_df.merge(
        result_df[["game_id", "home_team"]],
        left_on=["game_id", "team"],
        right_on=["game_id", "home_team"],
        how="inner",
    )
    
    rolling_feature_cols = [
        col for col in rolling_epa_df.columns
        if col not in ["game_id", "team"] and "last" in col
    ]
    
    home_rolling_map = {col: f"home_roll_epa_{col}" for col in rolling_feature_cols}
    home_rolling = home_rolling.rename(columns=home_rolling_map)
    
    result_df = result_df.merge(
        home_rolling[["game_id"] + [f"home_roll_epa_{col}" for col in rolling_feature_cols]],
        on="game_id",
        how="left",
    )
    
    # Merge rolling EPA features for away team
    logger.info("Merging away rolling EPA features")
    away_rolling = rolling_epa_df.merge(
        result_df[["game_id", "away_team"]],
        left_on=["game_id", "team"],
        right_on=["game_id", "away_team"],
        how="inner",
    )
    
    away_rolling_map = {col: f"away_roll_epa_{col}" for col in rolling_feature_cols}
    away_rolling = away_rolling.rename(columns=away_rolling_map)
    
    result_df = result_df.merge(
        away_rolling[["game_id"] + [f"away_roll_epa_{col}" for col in rolling_feature_cols]],
        on="game_id",
        how="left",
    )
    
    # Merge QB features for home team
    logger.info("Merging home QB features")
    home_qb = qb_features_df.merge(
        result_df[["game_id", "home_team"]],
        left_on=["game_id", "team"],
        right_on=["game_id", "home_team"],
        how="inner",
    )
    
    qb_feature_cols = [
        col for col in qb_features_df.columns
        if col not in ["game_id", "team"]
    ]
    
    home_qb_map = {col: f"home_qb_{col}" for col in qb_feature_cols}
    home_qb = home_qb.rename(columns=home_qb_map)
    
    result_df = result_df.merge(
        home_qb[["game_id"] + [f"home_qb_{col}" for col in qb_feature_cols]],
        on="game_id",
        how="left",
    )
    
    # Merge QB features for away team
    logger.info("Merging away QB features")
    away_qb = qb_features_df.merge(
        result_df[["game_id", "away_team"]],
        left_on=["game_id", "team"],
        right_on=["game_id", "away_team"],
        how="inner",
    )
    
    away_qb_map = {col: f"away_qb_{col}" for col in qb_feature_cols}
    away_qb = away_qb.rename(columns=away_qb_map)
    
    result_df = result_df.merge(
        away_qb[["game_id"] + [f"away_qb_{col}" for col in qb_feature_cols]],
        on="game_id",
        how="left",
    )
    
    # Validate
    logger.info("Validating merged features")
    
    # Check for duplicates
    if result_df["game_id"].duplicated().any():
        logger.error("Found duplicate game_ids in merged data!")
        raise ValueError("Duplicate game_ids found")
    
    # Check for missing features
    rolling_cols = [col for col in result_df.columns if "roll_epa" in col]
    qb_cols = [col for col in result_df.columns if "qb_" in col]
    
    missing_rolling = result_df[rolling_cols].isna().any(axis=1).sum()
    missing_qb = result_df[qb_cols].isna().any(axis=1).sum()
    
    if missing_rolling > 0:
        logger.warning(f"{missing_rolling} games missing rolling EPA features")
    if missing_qb > 0:
        logger.warning(f"{missing_qb} games missing QB features")
    
    # Sort by season, week, date
    result_df = result_df.sort_values(["season", "week", "date"]).reset_index(drop=True)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(output_path, index=False)
    logger.info(f"Saved Phase 2B game-level features to {output_path}")
    logger.info(f"Final feature table: {len(result_df)} games, {len(result_df.columns)} columns")
    
    return result_df


def run_phase2b_feature_pipeline() -> pd.DataFrame:
    """
    Run the complete Phase 2B feature pipeline (baseline + EPA + rolling EPA + QB).
    
    Returns:
        DataFrame with game-level features including all Phase 2B metrics
    """
    import time
    pipeline_start = time.time()
    
    logger.info("=" * 60)
    logger.info("Running Phase 2B Feature Pipeline (Baseline + EPA + Rolling EPA + QB)")
    logger.info("=" * 60)
    
    # Step 1: Ensure Phase 2 features exist
    from orchestration.pipelines.feature_pipeline import run_phase2_feature_pipeline
    
    step_start = time.time()
    logger.info("\n[Step 1/4] Ensuring Phase 2 features exist...")
    phase2_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "nfl"
        / "processed"
        / "game_features_phase2.parquet"
    )
    if not phase2_path.exists():
        logger.info("Phase 2 features not found, generating...")
        run_phase2_feature_pipeline()
        step_time = time.time() - step_start
        logger.info(f"✓ Phase 2 features generated ({step_time:.2f}s)")
    else:
        step_time = time.time() - step_start
        logger.info(f"✓ Phase 2 features already exist ({step_time:.2f}s)")
    
    # Step 2: Generate rolling EPA features
    from features.nfl.rolling_epa_features import generate_rolling_epa_features
    
    step_start = time.time()
    logger.info("\n[Step 2/4] Generating rolling EPA features...")
    rolling_epa_df = generate_rolling_epa_features()
    step_time = time.time() - step_start
    logger.info(f"✓ Generated rolling EPA features: {len(rolling_epa_df):,} team-games ({step_time:.2f}s)")
    
    rolling_feature_count = len([col for col in rolling_epa_df.columns 
                                if col not in ["game_id", "team"] and "last" in col])
    logger.info(f"  Added {rolling_feature_count} rolling EPA feature columns")
    
    # Step 3: Generate QB features
    from features.nfl.qb_features import generate_team_qb_features
    
    step_start = time.time()
    logger.info("\n[Step 3/4] Generating QB features...")
    qb_features_df = generate_team_qb_features()
    step_time = time.time() - step_start
    logger.info(f"✓ Generated QB features: {len(qb_features_df):,} team-games ({step_time:.2f}s)")
    
    qb_feature_count = len([col for col in qb_features_df.columns if col not in ["game_id", "team"]])
    logger.info(f"  Added {qb_feature_count} QB feature columns")
    
    # Step 4: Merge all features
    step_start = time.time()
    logger.info("\n[Step 4/4] Merging all features into Phase 2B table...")
    game_features_df = merge_phase2b_features()
    step_time = time.time() - step_start
    logger.info(f"✓ Created Phase 2B game-level features: {len(game_features_df):,} games ({step_time:.2f}s)")
    
    # Count total features
    total_feature_count = len([col for col in game_features_df.columns 
                              if col not in ["game_id", "season", "week", "date", "home_team", "away_team", 
                                            "home_score", "away_score", "close_spread", "close_total", 
                                            "open_spread", "open_total"]])
    logger.info(f"  Total features: {total_feature_count}")
    
    pipeline_time = time.time() - pipeline_start
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2B Feature Pipeline Complete!")
    logger.info(f"Total pipeline time: {pipeline_time:.2f} seconds")
    logger.info("=" * 60)
    
    return game_features_df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "phase2b":
        df = run_phase2b_feature_pipeline()
        print(f"\nFinal feature table shape: {df.shape}")
        print(f"\nRolling EPA columns: {[col for col in df.columns if 'roll_epa' in col][:10]}")
        print(f"\nQB columns: {[col for col in df.columns if 'qb_' in col][:10]}")
        print(f"\nSample features:")
        print(df[["game_id", "season", "week", "home_team", "away_team",
                  "home_roll_epa_off_epa_last3", "away_roll_epa_off_epa_last3",
                  "home_qb_qb_epa_per_dropback", "away_qb_qb_epa_per_dropback"]].head(10))
    elif len(sys.argv) > 1 and sys.argv[1] == "phase2":
        df = run_phase2_feature_pipeline()
        print(f"\nFinal feature table shape: {df.shape}")
        print(f"\nEPA feature columns: {[col for col in df.columns if 'epa' in col.lower()]}")
        print(f"\nSample features:")
        print(df[["game_id", "season", "week", "home_team", "away_team",
                  "home_epa_offensive_epa_per_play", "away_epa_offensive_epa_per_play",
                  "home_win_rate_last4", "away_win_rate_last4"]].head(10))
    else:
        df = run_baseline_feature_pipeline()
        print(f"\nFinal feature table shape: {df.shape}")
        print(f"\nFeature columns: {[col for col in df.columns if col.startswith(('home_', 'away_'))]}")
        print(f"\nSample features:")
        print(df[["game_id", "season", "week", "home_team", "away_team", 
                  "home_win_rate_last4", "away_win_rate_last4",
                  "home_pdiff_last4", "away_pdiff_last4"]].head(10))

