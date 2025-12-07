"""
Test Predictions for First 10 Games of 2025 Season

This script:
1. Loads the trained ensemble model from 2025 holdout evaluation
2. Predicts the first 10 games of 2025 season
3. Compares predictions against actual results
4. Tests for data leakage
5. Calculates accuracy metrics

Requirements:
- Trained model from evaluate_2025_holdout.py
- Actual game results in data/nfl/staged/games.parquet
- Features for 2025 games in data/nfl/processed/game_features_baseline.parquet
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.training.trainer import load_config
from models.architectures.stacking_ensemble import StackingEnsemble
from models.calibration import CalibratedModel
from eval.metrics import accuracy, brier_score, log_loss, calibration_buckets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_actual_results(game_id: str) -> Optional[Dict]:
    """
    Get actual game results from games.parquet.
    
    Args:
        game_id: Game ID
    
    Returns:
        Dictionary with actual results or None if not found
    """
    games_path = Path(__file__).parent.parent / "data" / "nfl" / "staged" / "games.parquet"
    
    if not games_path.exists():
        logger.warning(f"Games file not found: {games_path}")
        return None
    
    df = pd.read_parquet(games_path)
    game_data = df[df['game_id'] == game_id]
    
    if len(game_data) == 0:
        # Try alternative game_id formats
        # Original format: nfl_2025_WEEK_AWAY_HOME
        # Try: 2025_WK##_HOME_AWAY
        parts = game_id.split('_')
        if len(parts) >= 4 and parts[0] == 'nfl':
            season = parts[1]
            week = parts[2]
            away = parts[3]
            home = parts[4] if len(parts) > 4 else None
            # Try alternative format
            alt_id = f"{season}_WK{week}_{home}_{away}" if home else None
            if alt_id:
                game_data = df[df['game_id'] == alt_id]
        
        if len(game_data) == 0:
            logger.warning(f"Game {game_id} not found in games.parquet")
            return None
    
    row = game_data.iloc[0]
    
    return {
        'home_score': int(row['home_score']) if pd.notna(row['home_score']) else None,
        'away_score': int(row['away_score']) if pd.notna(row['away_score']) else None,
        'home_team': row['home_team'],
        'away_team': row['away_team'],
        'date': row.get('date', None),
        'game_id': row.get('game_id', game_id),
    }


def load_trained_model(model_dir: Path):
    """Load the trained ensemble model from holdout evaluation."""
    logger.info(f"Loading model from {model_dir}")
    
    # The ensemble is saved as a dict, need to reconstruct it
    ensemble_path = model_dir / "ensemble_v1.pkl"
    if not ensemble_path.exists():
        raise FileNotFoundError(
            f"Model not found at {ensemble_path}. "
            "Make sure evaluate_2025_holdout.py has been run successfully."
        )
    
    logger.info(f"Loading ensemble from {ensemble_path}")
    import pickle
    with open(ensemble_path, 'rb') as f:
        ensemble_dict = pickle.load(f)
    
    # Reconstruct StackingEnsemble from saved dict
    from models.architectures.stacking_ensemble import StackingEnsemble
    from models.architectures.ft_transformer import FTTransformerModel
    from models.architectures.tabnet import TabNetModel
    from models.architectures.gradient_boosting import GradientBoostingModel
    
    # Load base models
    base_models = {}
    base_model_paths = ensemble_dict.get('base_model_paths', {})
    model_names = ensemble_dict.get('model_names', [])
    
    for name in model_names:
        base_path = base_model_paths.get(name)
        base_path = Path(base_path) if base_path else None
        
        if base_path and base_path.exists():
            # Determine model type from path/name
            if 'ft_transformer' in name.lower() or 'ft_transformer' in str(base_path):
                base_models[name] = FTTransformerModel.load(base_path)
            elif 'tabnet' in name.lower() or 'tabnet' in str(base_path):
                # TabNet pickle contains path to .tabnet file, but actual file is .tabnet.zip
                # Load pickle, then fix the path
                import pickle
                with open(base_path, 'rb') as f:
                    tabnet_dict = pickle.load(f)
                # Fix model_path to point to .tabnet.zip
                if 'model_path' in tabnet_dict:
                    old_path = Path(tabnet_dict['model_path'])
                    zip_path = model_dir / f"{old_path.stem}.tabnet.zip"
                    if zip_path.exists():
                        tabnet_dict['model_path'] = str(zip_path)
                    else:
                        # Try tabnet.tabnet.zip
                        alt_zip = model_dir / "tabnet.tabnet.zip"
                        if alt_zip.exists():
                            tabnet_dict['model_path'] = str(alt_zip)
                # Create temporary pickle with fixed path
                import tempfile
                with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as tmp:
                    pickle.dump(tabnet_dict, tmp)
                    tmp_path = tmp.name
                try:
                    base_models[name] = TabNetModel.load(Path(tmp_path))
                finally:
                    Path(tmp_path).unlink()
            elif 'gbm' in name.lower() or 'gradient' in name.lower() or 'gbm' in str(base_path):
                base_models[name] = GradientBoostingModel.load(base_path)
            else:
                logger.warning(f"Unknown model type for {name}, trying generic load")
                from models.base import BaseModel
                base_models[name] = BaseModel.load(base_path)
        else:
            # Try loading from model_dir
            alt_path = model_dir / f"{name}.pkl"
            if alt_path.exists():
                if 'ft_transformer' in name.lower():
                    base_models[name] = FTTransformerModel.load(alt_path)
                elif 'tabnet' in name.lower():
                    # Try .pkl first, then .tabnet.zip
                    base_models[name] = TabNetModel.load(alt_path)
                elif 'gbm' in name.lower() or 'gradient' in name.lower():
                    base_models[name] = GradientBoostingModel.load(alt_path)
                else:
                    from models.base import BaseModel
                    base_models[name] = BaseModel.load(alt_path)
            else:
                # Try tabnet.zip if name is tabnet
                if 'tabnet' in name.lower():
                    zip_path = model_dir / "tabnet.tabnet.zip"
                    if zip_path.exists():
                        base_models[name] = TabNetModel.load(zip_path)
                    else:
                        zip_path = model_dir / f"{name}.tabnet.zip"
                        if zip_path.exists():
                            base_models[name] = TabNetModel.load(zip_path)
    
    # Reconstruct ensemble
    config = ensemble_dict.get('config', {})
    meta_cfg = config.get('meta_model', {})
    stack_cfg = config.get('stacking', {})
    
    ensemble = StackingEnsemble(
        base_models=base_models,
        meta_model_type=meta_cfg.get('type', 'logistic'),
        include_features=stack_cfg.get('include_features', False),
        feature_fraction=stack_cfg.get('feature_fraction', 0.0),
        mlp_hidden_dims=meta_cfg.get('mlp_hidden_dims', [16, 8]),
        mlp_dropout=meta_cfg.get('mlp_dropout', 0.1),
        mlp_epochs=meta_cfg.get('mlp_epochs', 50),
        random_state=config.get('random_state', 42),
    )
    
    # Add base models to ensemble (they should already be in base_models dict passed to constructor)
    # But ensure they're properly set
    ensemble.base_models = base_models
    ensemble.model_names = model_names
    ensemble.n_base_models = len(base_models)
    
    # Restore meta-model and scaler
    ensemble.meta_model = ensemble_dict.get('meta_model')
    ensemble.scaler = ensemble_dict.get('scaler')
    ensemble.selected_feature_indices = ensemble_dict.get('selected_feature_indices')
    
    # Verify base models loaded
    logger.info(f"✓ Model reconstructed: {type(ensemble).__name__}")
    logger.info(f"  Base models loaded: {list(base_models.keys())}")
    logger.info(f"  Model names: {model_names}")
    logger.info(f"  Meta-model type: {ensemble.meta_model_type}")
    logger.info(f"  Meta-model: {type(ensemble.meta_model).__name__ if ensemble.meta_model else 'None'}")
    
    # Check for calibration
    calibration_path = model_dir / "calibration.pkl"
    if calibration_path.exists():
        logger.info("Loading calibration...")
        with open(calibration_path, 'rb') as f:
            calibration = pickle.load(f)
        from models.calibration import CalibratedModel
        model = CalibratedModel(ensemble, calibration)
        logger.info("✓ Calibration applied")
    else:
        model = ensemble
    
    return model


def prepare_features_for_prediction(game_data: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    Prepare features for prediction from game data.
    Uses the same feature selection logic as trainer.py load_features().
    
    Args:
        game_data: DataFrame with game features
    
    Returns:
        Tuple of (feature matrix, feature column names)
    """
    # Use same exclude columns as trainer.py load_features()
    exclude_cols = [
        "game_id",
        "season",
        "week",
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "home_win",  # Target
        "close_spread",  # Keep for ROI calculation but not as feature
        "close_total",  # Keep for ROI calculation but not as feature
        "open_spread",  # Optional, exclude from features
        "open_total",  # Optional, exclude from features
    ]
    
    feature_cols = [col for col in game_data.columns if col not in exclude_cols]
    
    # Select features
    X = game_data[feature_cols].copy()
    
    # Handle missing values (same as trainer.py)
    X = X.fillna(0)
    
    return X, feature_cols


def check_for_leakage(game_data: pd.DataFrame, feature_cols: List[str]) -> Dict[str, bool]:
    """
    Check for potential data leakage in features.
    
    Args:
        game_data: Game data DataFrame
        feature_cols: List of feature column names
    
    Returns:
        Dictionary of leakage checks
    """
    leakage_checks = {
        'has_future_scores': False,
        'has_result_column': False,
        'has_spread_column': False,
        'has_future_stats': False,
    }
    
    # Check for score columns
    if 'home_score' in game_data.columns and pd.notna(game_data['home_score']).any():
        leakage_checks['has_future_scores'] = True
    
    if 'result' in game_data.columns:
        leakage_checks['has_result_column'] = True
    
    if 'spread' in game_data.columns:
        leakage_checks['has_spread_column'] = True
    
    # Check for future-looking feature names
    future_keywords = ['final', 'post', 'end', 'total', 'final_score', 'result']
    for col in feature_cols:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in future_keywords):
            if 'future' not in leakage_checks:
                leakage_checks['has_future_stats'] = True
                leakage_checks['future_feature'] = col
            break
    
    return leakage_checks


def predict_spread(prob: float) -> float:
    """Convert probability to estimated spread."""
    # Simple linear mapping: prob 0.5 = spread 0, prob 1.0 = spread +14, prob 0.0 = spread -14
    return (prob - 0.5) * 28


def main():
    """Main function."""
    logger.info("=" * 80)
    logger.info("Testing Predictions for First 10 Games of 2025 Season")
    logger.info("=" * 80)
    
    # Paths
    project_root = Path(__file__).parent.parent
    model_dir = project_root / "artifacts" / "models" / "nfl_stacked_ensemble_2025_holdout"
    features_path = project_root / "data" / "nfl" / "processed" / "game_features_baseline.parquet"
    games_path = project_root / "data" / "nfl" / "staged" / "games.parquet"
    
    # Check files exist
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not games_path.exists():
        raise FileNotFoundError(f"Games file not found: {games_path}")
    
    # Load model
    logger.info("\n[Step 1/6] Loading trained model...")
    model = load_trained_model(model_dir)
    logger.info("✓ Model loaded")
    
    # Load features
    logger.info("\n[Step 2/6] Loading features...")
    features_df = pd.read_parquet(features_path)
    logger.info(f"Loaded {len(features_df)} total games")
    
    # Filter to first 10 games of 2025 season
    logger.info("\n[Step 3/6] Filtering to first 10 games of 2025...")
    season_2025 = features_df[features_df['season'] == 2025].copy()
    season_2025 = season_2025.sort_values(['week', 'game_id'])
    
    if len(season_2025) == 0:
        raise ValueError("No 2025 games found in features!")
    
    first_10_games = season_2025.head(10).copy()
    logger.info(f"Found {len(first_10_games)} games (first 10 of 2025)")
    logger.info(f"Weeks: {sorted(first_10_games['week'].unique())}")
    
    # Check for leakage
    logger.info("\n[Step 4/6] Checking for data leakage...")
    leakage_results = []
    for idx, row in first_10_games.iterrows():
        game_data = pd.DataFrame([row])
        _, feature_cols = prepare_features_for_prediction(game_data)
        leakage = check_for_leakage(game_data, feature_cols)
        leakage['game_id'] = row['game_id']
        leakage_results.append(leakage)
    
    leakage_df = pd.DataFrame(leakage_results)
    has_leakage = leakage_df[['has_future_scores', 'has_result_column', 'has_spread_column', 'has_future_stats']].any().any()
    
    if has_leakage:
        logger.warning("⚠ WARNING: Potential data leakage detected!")
        logger.warning(leakage_df[leakage_df[['has_future_scores', 'has_result_column', 'has_spread_column', 'has_future_stats']].any(axis=1)])
    else:
        logger.info("✓ No data leakage detected")
    
    # Make predictions
    logger.info("\n[Step 5/6] Making predictions...")
    predictions = []
    
    for idx, row in first_10_games.iterrows():
        game_id = row['game_id']
        home_team = row['home_team']
        away_team = row['away_team']
        week = row['week']
        
        # Prepare features
        game_data = pd.DataFrame([row])
        X, feature_cols = prepare_features_for_prediction(game_data)
        
        # Predict
        prob = model.predict_proba(X)[0]
        predicted_winner = home_team if prob >= 0.5 else away_team
        predicted_spread = predict_spread(prob)
        confidence = max(prob, 1 - prob)
        
        # Get actual results
        actual_results = get_actual_results(game_id)
        
        actual_winner = None
        actual_home_score = None
        actual_away_score = None
        actual_spread = None
        correct = None
        
        if actual_results and actual_results.get('home_score') is not None:
            actual_home_score = int(actual_results['home_score'])
            actual_away_score = int(actual_results['away_score'])
            actual_winner = actual_results['home_team'] if actual_home_score > actual_away_score else actual_results['away_team']
            actual_spread = actual_home_score - actual_away_score
            correct = (predicted_winner == actual_winner)
        
        predictions.append({
            'game_id': game_id,
            'week': week,
            'home_team': home_team,
            'away_team': away_team,
            'predicted_winner': predicted_winner,
            'predicted_prob': prob,
            'predicted_spread': predicted_spread,
            'confidence': confidence,
            'actual_winner': actual_winner,
            'actual_home_score': actual_home_score,
            'actual_away_score': actual_away_score,
            'actual_spread': actual_spread,
            'correct': correct,
        })
        
        logger.info(f"  Week {week}: {away_team} @ {home_team}")
        logger.info(f"    Prediction: {predicted_winner} ({prob:.3f}, spread: {predicted_spread:+.1f})")
        if actual_winner:
            logger.info(f"    Actual: {actual_winner} (spread: {actual_spread:+.1f}) {'✓' if correct else '✗'}")
        else:
            logger.info(f"    Actual: Not available")
    
    predictions_df = pd.DataFrame(predictions)
    
    # Calculate metrics
    logger.info("\n[Step 6/6] Calculating accuracy metrics...")
    
    completed_games = predictions_df[predictions_df['correct'].notna()].copy()
    
    if len(completed_games) == 0:
        logger.warning("No completed games found - cannot calculate accuracy")
        logger.info("Predictions made, but actual results not available")
        return
    
    # Accuracy metrics
    n_correct = completed_games['correct'].sum()
    n_total = len(completed_games)
    accuracy_pct = (n_correct / n_total) * 100
    
    # Spread error
    spread_errors = np.abs(completed_games['predicted_spread'] - completed_games['actual_spread'])
    spread_mae = spread_errors.mean()
    
    # Probability metrics (if we have actual binary outcomes)
    actual_binary = (completed_games['actual_winner'] == completed_games['home_team']).astype(int)
    predicted_probs = completed_games['predicted_prob'].values
    
    brier = brier_score(actual_binary, predicted_probs)
    logloss = log_loss(actual_binary, predicted_probs)
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION TEST RESULTS - First 10 Games of 2025")
    logger.info("=" * 80)
    logger.info(f"\nGames Analyzed: {len(predictions_df)}")
    logger.info(f"Games with Results: {len(completed_games)}")
    logger.info(f"\nAccuracy Metrics:")
    logger.info(f"  Correct Predictions: {n_correct} / {n_total}")
    logger.info(f"  Accuracy: {accuracy_pct:.1f}%")
    logger.info(f"\nSpread Metrics:")
    logger.info(f"  Mean Absolute Error: {spread_mae:.2f} points")
    logger.info(f"\nProbability Metrics:")
    logger.info(f"  Brier Score: {brier:.4f}")
    logger.info(f"  Log Loss: {logloss:.4f}")
    
    # Detailed results
    logger.info(f"\nDetailed Results:")
    logger.info("-" * 80)
    for _, row in completed_games.iterrows():
        status = "✓" if row['correct'] else "✗"
        logger.info(f"{status} Week {row['week']:2d}: {row['away_team']:3s} @ {row['home_team']:3s} | "
                   f"Pred: {row['predicted_winner']:3s} ({row['predicted_prob']:.3f}, {row['predicted_spread']:+5.1f}) | "
                   f"Actual: {row['actual_winner']:3s} ({row['actual_spread']:+5.1f})")
    
    # Leakage summary
    logger.info(f"\nLeakage Check Summary:")
    logger.info(f"  Future scores in features: {leakage_df['has_future_scores'].any()}")
    logger.info(f"  Result column present: {leakage_df['has_result_column'].any()}")
    logger.info(f"  Spread column present: {leakage_df['has_spread_column'].any()}")
    logger.info(f"  Future-looking features: {leakage_df['has_future_stats'].any()}")
    
    if not has_leakage:
        logger.info("  ✓ No data leakage detected")
    else:
        logger.warning("  ⚠ Potential leakage detected - review feature engineering")
    
    # Save results
    output_dir = project_root / "artifacts" / "predictions" / "first_10_games_2025"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "predictions_vs_actuals.parquet"
    predictions_df.to_parquet(output_path, index=False)
    logger.info(f"\n✓ Results saved to {output_path}")
    
    # Save summary report
    report_path = output_dir / "test_report.md"
    with open(report_path, 'w') as f:
        f.write("# Prediction Test Report - First 10 Games of 2025\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- **Games Analyzed**: {len(predictions_df)}\n")
        f.write(f"- **Games with Results**: {len(completed_games)}\n")
        f.write(f"- **Accuracy**: {accuracy_pct:.1f}% ({n_correct}/{n_total})\n")
        f.write(f"- **Spread MAE**: {spread_mae:.2f} points\n")
        f.write(f"- **Brier Score**: {brier:.4f}\n")
        f.write(f"- **Log Loss**: {logloss:.4f}\n\n")
        f.write(f"## Leakage Check\n\n")
        f.write(f"- **Future scores in features**: {leakage_df['has_future_scores'].any()}\n")
        f.write(f"- **Result column present**: {leakage_df['has_result_column'].any()}\n")
        f.write(f"- **Spread column present**: {leakage_df['has_spread_column'].any()}\n")
        f.write(f"- **Future-looking features**: {leakage_df['has_future_stats'].any()}\n\n")
        f.write(f"## Detailed Results\n\n")
        f.write("| Week | Away | Home | Predicted | Prob | Spread | Actual | Spread | Correct |\n")
        f.write("|------|------|------|-----------|------|--------|--------|--------|---------|\n")
        for _, row in completed_games.iterrows():
            status = "✓" if row['correct'] else "✗"
            f.write(f"| {row['week']} | {row['away_team']} | {row['home_team']} | "
                   f"{row['predicted_winner']} | {row['predicted_prob']:.3f} | "
                   f"{row['predicted_spread']:+.1f} | {row['actual_winner']} | "
                   f"{row['actual_spread']:+.1f} | {status} |\n")
    
    logger.info(f"✓ Report saved to {report_path}")
    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()

